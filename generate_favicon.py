"""
Script to generate favicon.ico from mysitelogo.png
"""

from PIL import Image
import os

def generate_favicon():
    """Generate favicon.ico from mysitelogo.png"""
    try:
        # Check if source image exists
        source_path = "mysitelogo1.png"
        if not os.path.exists(source_path):
            print(f"Source image not found: {source_path}")
            print("Please ensure mysitelogo.png is in the web directory")
            return False
        
        # Open and resize the image
        print("Opening source image...")
        img = Image.open(source_path)
        
        # Convert to RGBA if not already (for transparency support)
        if img.mode != 'RGBA':
            img = img.convert('RGBA')
        
        # Resize to 32x32 pixels
        print("Resizing image to 32x32...")
        img = img.resize((32, 32), Image.Resampling.LANCZOS)
        
        # Save as ICO file
        output_path = "web/static/favicon.ico"
        print(f"Saving favicon to: {output_path}")
        img.save(output_path, format="ICO")
        
        print("Favicon generated successfully!")
        return True
        
    except ImportError:
        print("PIL/Pillow not installed. Installing...")
        os.system("pip install Pillow")
        print("Please run this script again after installation.")
        return False
    except Exception as e:
        print(f"Error generating favicon: {e}")
        return False

if __name__ == "__main__":
    generate_favicon()