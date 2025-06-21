import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import io

# Set page config
st.set_page_config(
    page_title="Handwritten Digit Generator",
    page_icon="🔢",
    layout="wide"
)

# UPDATED Model architecture (same as training)
class ImprovedConditionalVAE(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=512, latent_dim=32, num_classes=10):
        super(ImprovedConditionalVAE, self).__init__()
        
        # Encoder
        self.fc1 = nn.Linear(input_dim + num_classes, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, hidden_dim // 4)
        self.fc_mu = nn.Linear(hidden_dim // 4, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim // 4, latent_dim)
        
        # Decoder
        self.fc4 = nn.Linear(latent_dim + num_classes, hidden_dim // 4)
        self.fc5 = nn.Linear(hidden_dim // 4, hidden_dim // 2)
        self.fc6 = nn.Linear(hidden_dim // 2, hidden_dim)
        self.fc7 = nn.Linear(hidden_dim, input_dim)
        
        self.dropout = nn.Dropout(0.2)
        
    def encode(self, x, c):
        inputs = torch.cat([x, c], 1)
        h1 = F.relu(self.fc1(inputs))
        h1 = self.dropout(h1)
        h2 = F.relu(self.fc2(h1))
        h2 = self.dropout(h2)
        h3 = F.relu(self.fc3(h2))
        return self.fc_mu(h3), self.fc_logvar(h3)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z, c):
        inputs = torch.cat([z, c], 1)
        h4 = F.relu(self.fc4(inputs))
        h4 = self.dropout(h4)
        h5 = F.relu(self.fc5(h4))
        h5 = self.dropout(h5)
        h6 = F.relu(self.fc6(h5))
        return torch.sigmoid(self.fc7(h6))
    
    def forward(self, x, c):
        mu, logvar = self.encode(x.view(-1, 784), c)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, c), mu, logvar

@st.cache_resource
def load_model():
    device = torch.device('cpu')
    model = ImprovedConditionalVAE(latent_dim=32, hidden_dim=512)
    
    try:
        # Try to load the improved model first
        model.load_state_dict(torch.load('improved_digit_generator_model.pth', map_location=device))
    except:
        # Fallback to original model if improved version not available
        model.load_state_dict(torch.load('digit_generator_model.pth', map_location=device))
    
    model.eval()
    return model, device

def generate_digits(model, device, digit, num_samples=5, latent_dim=32):
    with torch.no_grad():
        # Create one-hot encoded label for the digit
        label = torch.zeros(num_samples, 10).to(device)
        label[:, digit] = 1
        
        # Sample from latent space with controlled randomness
        z = torch.randn(num_samples, latent_dim).to(device)
        
        # Generate images
        generated = model.decode(z, label)
        generated = generated.view(num_samples, 28, 28)
        
        # Ensure proper range [0, 1]
        generated = torch.clamp(generated, 0, 1)
        
    return generated.cpu().numpy()

def main():
    st.title("🔢 Improved Handwritten Digit Generator")
    st.write("Generate synthetic MNIST-like images using an improved Conditional VAE")
    
    # Load model
    try:
        model, device = load_model()
        st.success("✅ Model loaded successfully!")
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        st.stop()
    
    # User input
    st.subheader("Choose a digit to generate (0-9):")
    selected_digit = st.selectbox(
        "Select digit:",
        options=list(range(10)),
        index=2,
        help="Choose which digit you want to generate"
    )
    
    # Generate button
    if st.button("🎲 Generate Images", type="primary"):
        with st.spinner("Generating handwritten digits..."):
            try:
                # Generate 5 images
                generated_images = generate_digits(model, device, selected_digit, num_samples=5)
                
                st.subheader(f"Generated images of digit {selected_digit}")
                
                # Display images in a row
                cols = st.columns(5)
                for i in range(5):
                    with cols[i]:
                        # Convert to PIL Image for better display
                        img_array = (generated_images[i] * 255).astype(np.uint8)
                        img = Image.fromarray(img_array, mode='L')
                        
                        # Resize for better visibility
                        img_resized = img.resize((112, 112), Image.NEAREST)
                        
                        st.image(img_resized, caption=f"Sample {i+1}", use_column_width=True)
                
                st.success("✨ Successfully generated 5 unique images!")
                
            except Exception as e:
                st.error(f"❌ Error generating images: {str(e)}")
    
    # Information section
    st.markdown("---")
    st.subheader("ℹ️ About this Improved App")
    st.write("""
    This web application uses an **Improved Conditional Variational Autoencoder (CVAE)** with:
    
    **Key Improvements:**
    - Deeper architecture with 512 hidden units
    - Better data preprocessing (no problematic normalization)
    - Proper loss function weighting (Beta-VAE)
    - Dropout for regularization
    - 50 training epochs for better convergence
    
    **Technical Details:**
    - Model: Improved Conditional VAE with 32-dimensional latent space
    - Training: Google Colab with T4 GPU (50 epochs)
    - Framework: PyTorch + Streamlit
    """)

if __name__ == "__main__":
    main()
