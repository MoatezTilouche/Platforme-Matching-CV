# run_api.py
"""
Startup script for CV-Job Matching API
"""
import uvicorn
import sys
from pathlib import Path

def check_ollama():
    """Check if Ollama models are available"""
    import subprocess
    
    try:
        result = subprocess.run(
            ["ollama", "list"], 
            capture_output=True, 
            text=True,
            timeout=5
        )
        
        models_output = result.stdout
        
        required_models = {
            "mistral:7b-instruct": False
        }
        
        for model in required_models:
            if model in models_output:
                required_models[model] = True
        
        missing = [m for m, found in required_models.items() if not found]
        
        if missing:
            print("\n⚠️  Warning: Missing Ollama models:")
            for model in missing:
                print(f"   • {model}")
            print("\nTo install missing models, run:")
            for model in missing:
                print(f"   ollama pull {model}")
            print()
            
            response = input("Continue anyway? (y/n): ")
            if response.lower() != 'y':
                print("Exiting...")
                sys.exit(1)
        else:
            print("✅ All required Ollama models are installed")
            
    except FileNotFoundError:
        print("⚠️  Warning: Ollama not found in PATH")
        print("Make sure Ollama is installed and running")
        print()
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            print("Exiting...")
            sys.exit(1)
    except Exception as e:
        print(f"⚠️  Could not check Ollama: {e}")


def main():
    print("\n" + "="*60)
    print("🚀 CV-Job Matching API Server")
    print("="*60)
    
    # Check Ollama
    print("\n📦 Checking prerequisites...")
    check_ollama()
    
    # Server config
    host = "0.0.0.0"
    port = 8000
    
    print("\n" + "="*60)
    print("🌐 Starting API server...")
    print("="*60)
    print(f"\n📍 Local:    http://localhost:{port}")
    print(f"📍 Network:  http://{host}:{port}")
    print(f"📚 API Docs: http://localhost:{port}/docs")
    print(f"📊 ReDoc:    http://localhost:{port}/redoc")
    print("\n" + "="*60)
    print("\n⚡ Ready to accept requests!\n")
    
    # Start server
    uvicorn.run(
        "app.main:app",
        host=host,
        port=port,
        reload=True,
        log_level="info"
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Shutting down server...")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)
