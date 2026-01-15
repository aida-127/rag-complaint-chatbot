import gradio as gr
import chromadb
from chromadb.config import Settings

# Initialize
print("Loading vector store...")
try:
    client = chromadb.PersistentClient(path="vector_store")
    collection = client.get_collection("complaints")
    print(f"Loaded {collection.count()} complaints")
except:
    print("ERROR: Vector store not found")
    collection = None

def rag_response(question):
    if not question.strip():
        return "Please enter a question."
    
    if not collection:
        return "Vector store not available. Run Task 2 first."
    
    # Retrieve
    results = collection.query(
        query_texts=[question],
        n_results=3,
        include=["documents", "metadatas"]
    )
    
    # Format answer
    answer = f"**Found {len(results['documents'][0])} relevant complaints:**\n\n"
    
    for i, (doc, meta) in enumerate(zip(results['documents'][0], results['metadatas'][0]), 1):
        product = meta.get('product', 'Unknown')
        answer += f"{i}. **{product}**: {doc[:150]}...\n\n"
    
    return answer

# Create interface
with gr.Blocks() as app:
    gr.Markdown("# Complaint Analysis Chatbot")
    gr.Markdown("Ask questions about customer complaints")
    
    with gr.Row():
        inp = gr.Textbox(label="Your question", placeholder="e.g., credit card issues")
        out = gr.Markdown(label="Answer")
    
    btn = gr.Button("Submit")
    btn.click(fn=rag_response, inputs=inp, outputs=out)
    
    inp.submit(fn=rag_response, inputs=inp, outputs=out)

if __name__ == "__main__":
    print("Launching app...")
    app.launch(server_name="127.0.0.1", server_port=7860, share=False, debug=True)
    print("App closed")
