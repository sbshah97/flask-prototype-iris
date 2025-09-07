#!/usr/bin/env python3
"""
Setup script for NITK Academic Advisor
Helps users get started quickly with the right environment setup.
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

def check_python_version():
    """Check if Python version is 3.11+"""
    version = sys.version_info
    if version < (3, 11):
        print(f"❌ Python 3.11+ required, found {version.major}.{version.minor}")
        print("Please upgrade Python and try again")
        return False
    
    print(f"✅ Python {version.major}.{version.minor} detected")
    return True

def check_git():
    """Check if git is available"""
    if shutil.which("git"):
        print("✅ Git is available")
        return True
    else:
        print("❌ Git not found - needed for version control")
        return False

def create_virtual_environment():
    """Create and activate virtual environment"""
    print("🔍 Creating virtual environment...")
    
    venv_path = Path("venv")
    if venv_path.exists():
        print("⚠️  Virtual environment already exists")
        return True
    
    try:
        subprocess.run([sys.executable, "-m", "venv", "venv"], check=True)
        print("✅ Virtual environment created")
        
        # Detect activation script
        if sys.platform == "win32":
            activate_script = "venv\\Scripts\\activate"
        else:
            activate_script = "source venv/bin/activate"
        
        print(f"💡 Activate with: {activate_script}")
        return True
        
    except subprocess.CalledProcessError:
        print("❌ Failed to create virtual environment")
        return False

def install_dependencies():
    """Install Python dependencies"""
    print("🔍 Installing dependencies...")
    
    # Use the venv python if available
    if sys.platform == "win32":
        python_cmd = "venv\\Scripts\\python"
    else:
        python_cmd = "venv/bin/python"
    
    # Fall back to current python if venv not found
    if not Path(python_cmd).exists():
        python_cmd = sys.executable
    
    try:
        subprocess.run([
            python_cmd, "-m", "pip", "install", "--upgrade", "pip"
        ], check=True)
        
        subprocess.run([
            python_cmd, "-m", "pip", "install", "-r", "requirements.txt"
        ], check=True)
        
        print("✅ Dependencies installed successfully")
        return True
        
    except subprocess.CalledProcessError:
        print("❌ Failed to install dependencies")
        print("Try manually: pip install -r requirements.txt")
        return False

def setup_environment_file():
    """Create .env file from template"""
    print("🔍 Setting up environment file...")
    
    env_file = Path(".env")
    env_example = Path(".env.example")
    
    if env_file.exists():
        print("⚠️  .env file already exists")
        return True
    
    if not env_example.exists():
        print("❌ .env.example not found")
        return False
    
    # Copy template
    shutil.copy(env_example, env_file)
    print("✅ Created .env file from template")
    
    print("🔧 Please edit .env file and add your API keys:")
    print("   - GOOGLE_API_KEY (required for Gemini)")
    print("   - OPENAI_API_KEY (optional fallback)")
    
    return True

def create_directories():
    """Create required directories"""
    print("🔍 Creating required directories...")
    
    dirs = [
        "data/pdfs",
        "data/faiss_index",
        "logs"
    ]
    
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    print("✅ Directories created")
    return True

def test_imports():
    """Test that key modules can be imported"""
    print("🔍 Testing imports...")
    
    test_modules = [
        ("flask", "Flask"),
        ("langchain", "LangChain"),
        ("sentence_transformers", "SentenceTransformers"),
        ("faiss", "FAISS")
    ]
    
    failed = []
    for module, name in test_modules:
        try:
            __import__(module)
            print(f"✅ {name} available")
        except ImportError:
            print(f"❌ {name} not available")
            failed.append(name)
    
    if failed:
        print(f"❌ Missing modules: {', '.join(failed)}")
        print("Run: pip install -r requirements.txt")
        return False
    
    return True

def show_next_steps():
    """Show what to do next"""
    print("\n" + "=" * 50)
    print("🎉 Setup Complete!")
    print("=" * 50)
    
    print("\n📋 Next Steps:")
    print("1. Edit .env file with your API keys")
    print("2. Get Google AI API key from: https://aistudio.google.com/app/apikey")
    print("3. Run the application:")
    
    if sys.platform == "win32":
        print("   venv\\Scripts\\activate")
    else:
        print("   source venv/bin/activate")
    
    print("   python app.py")
    print("4. Open http://localhost:8000 in your browser")
    print("5. Test with: python test_app.py")
    
    print("\n📚 Documentation:")
    print("- README.md - Full documentation")
    print("- .env.example - Environment variable reference")
    
    print("\n🚀 Deployment:")
    print("- Railway: Connect your GitHub repo")
    print("- Set environment variables in Railway dashboard")

def main():
    """Run setup process"""
    print("🚀 NITK Academic Advisor - Setup")
    print("=" * 40)
    
    steps = [
        ("Python Version Check", check_python_version),
        ("Git Check", check_git),
        ("Virtual Environment", create_virtual_environment),
        ("Install Dependencies", install_dependencies),
        ("Environment File", setup_environment_file),
        ("Create Directories", create_directories),
        ("Test Imports", test_imports)
    ]
    
    failed_steps = []
    for step_name, step_func in steps:
        print(f"\n📋 {step_name}...")
        try:
            if not step_func():
                failed_steps.append(step_name)
        except Exception as e:
            print(f"❌ {step_name} failed: {e}")
            failed_steps.append(step_name)
    
    if failed_steps:
        print(f"\n❌ Setup incomplete. Failed steps: {', '.join(failed_steps)}")
        print("Please resolve the issues above and run setup again.")
        sys.exit(1)
    else:
        show_next_steps()

if __name__ == "__main__":
    main()