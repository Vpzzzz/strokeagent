import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from rag import (
    load_and_chunk_pdfs, 
    setup_hybrid_retriever, 
    get_pdf_master_list, 
    setup_rag_agent,
    PDF_DIR, CHROMA_PATH
)
import os

# --- 0. Streamlit App Configuration ---
st.set_page_config(page_title="Hybrid RAG Agent Chat")
st.title("🤖 Hybrid RAG Agent (Ollama + RRF + Agentic)")

# Use a consistent thread ID for the entire session
THREAD_ID = "streamlit_session_1"

# --- 1. Initialization and Caching ---

@st.cache_resource
def initialize_agent():
    """Initializes the expensive RAG components once."""
    
    chunks = load_and_chunk_pdfs(PDF_DIR)
    is_db_exists = os.path.exists(CHROMA_PATH) and len(os.listdir(CHROMA_PATH)) > 0

    if not chunks and not is_db_exists:
        st.error("Setup Failed: No documents found to load or index.")
        return None
        
    pdf_master_list = get_pdf_master_list(PDF_DIR)
    final_retriever = setup_hybrid_retriever(chunks)
    agent = setup_rag_agent(final_retriever, pdf_master_list)
    
    st.info(f"RAG Agent Setup Complete. Knowledge Base:\n{pdf_master_list}")
    
    return agent

agent = initialize_agent()

# Initialize chat history in Streamlit's session state
if "messages" not in st.session_state:
    st.session_state.messages = [
        AIMessage(content="Hello! I am a Hybrid RAG Agent. Ask me questions about the documents."),
    ]
    
# --- 2. Display Chat History ---
for message in st.session_state.messages:
    role = "user" if isinstance(message, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.markdown(message.content)

# --- 3. Handle User Input and Agent Invocation (Streaming with Thinking) ---

if prompt := st.chat_input("Ask a question about the PDFs..."):
    if agent is None:
        st.error("Agent is not initialized. Check console for setup errors.")
        st.stop()

    # Display user message
    st.session_state.messages.append(HumanMessage(content=prompt))
    with st.chat_message("user"):
        st.markdown(prompt)

    # Setup the display areas for streaming
    response_container = st.container()
    full_response = ""
    thinking_log = "🧠 **Agent Thinking Process**\n\n"
    
    # Display placeholders for thinking and answer within the container
    with response_container:
        # Collapsible section for the thinking log (expanded by default for visibility)
        with st.expander("🔍 Agent Thinking (Tool Calls & Observations)", expanded=True) as thought_expander:
             thought_placeholder = st.empty()
        # Main area for the streaming answer
        answer_placeholder = st.empty()
    
    try:
        # Use agent.stream() to access the thinking steps (node outputs)
        stream = agent.stream(
            {"messages": [HumanMessage(content=prompt)]},
            {"configurable": {"thread_id": THREAD_ID}}
        )

        for chunk in stream:
            # Iterate through all node updates in the chunk
            for node_name, data in chunk.items():
                
                # --- Capturing Message-Based Events (Tool Call & Final Answer) ---
                if 'messages' in data:
                    last_message = data['messages'][-1]
                    
                    # 1. Capture Thinking: Tool Call (Action decision by LLM)
                    # Check for tool_calls safely using hasattr on the message object
                    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
                         # This is the Agent's decision (an AIMessage)
                         tool_call = last_message.tool_calls[0]
                         tool_name = tool_call.get('name', 'document_search')
                         thought_query = tool_call.get('args', {}).get('query', 'N/A')
                         
                         thinking_log += f"**Action:** Called `{tool_name}`\n"
                         thinking_log += f"**Query:** `{thought_query}`\n"
                         thought_placeholder.markdown(thinking_log)
                         
                    # 2. Capture Final Answer Tokens (Answer Generation)
                    elif last_message.content and node_name == 'model':
                        content_chunk = last_message.content
                        
                        full_response += content_chunk
                        answer_placeholder.markdown(full_response + "▌") 
                        
                # --- Capturing Node-Based Events (Tool Execution) ---
                elif node_name == 'action':
                    # This node executes the tool and contains the observation
                    tool_output = data.get('output', 'No Output')
                    
                    # Handle observation display
                    if isinstance(tool_output, list) and tool_output:
                        # Display snippet from the first retrieved document's content
                        obs_snippet = str(tool_output[0].page_content)[:200].replace('\n', ' ') + "..."
                    else:
                        obs_snippet = str(tool_output)[:200].replace('\n', ' ') + "..."
                        
                    thinking_log += f"**Observation:** Retrieved documents (Snippet: *{obs_snippet}*)\n\n"
                    thought_placeholder.markdown(thinking_log)
                        
        # Final cleanup and display
        answer_placeholder.markdown(full_response)
        thought_placeholder.markdown(thinking_log + "---") # Finalize thinking log
            
    except Exception as e:
        full_response = f"An error occurred during agent execution: {e}"
        st.error(full_response)
        
    # --- 4. Update Session State ---
    final_agent_message = AIMessage(content=full_response)
    st.session_state.messages.append(final_agent_message)
