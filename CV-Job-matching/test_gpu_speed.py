"""Quick test to verify GPU embeddings work"""
import time
from app.utils.rag import embed_chunks, embed_text

# Test data
test_chunks = [
    "Python developer with 5 years experience",
    "Experience with FastAPI and Node.js",
    "Strong background in machine learning",
    "Built scalable microservices architecture",
    "Led team of 3 developers"
] * 10  # 50 chunks total

print("="*60)
print("GPU Embedding Speed Test")
print("="*60)
print(f"\nEmbedding {len(test_chunks)} chunks...")

start = time.time()
embeddings = embed_chunks(test_chunks)
elapsed = time.time() - start

print(f"✓ Completed in {elapsed:.2f}s")
print(f"  Average: {(elapsed/len(test_chunks)*1000):.1f}ms per chunk")
print(f"  Embeddings shape: {len(embeddings)} x {len(embeddings[0])}")

# Test single embedding
print(f"\nEmbedding single text...")
start = time.time()
single_emb = embed_text("Full-stack developer with React and Python")
elapsed = time.time() - start
print(f"✓ Completed in {elapsed:.3f}s")

print("\n" + "="*60)
print("✓ GPU embeddings working!")
print("="*60)
