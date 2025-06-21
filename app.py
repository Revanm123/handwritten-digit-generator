import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image

st.set_page_config(page_title="Handwritten Digit Generator", page_icon="🔢", layout="wide")

# IMPROVED Model architecture (must match your training exactly)
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
        
    def decode(self, z, c):
        inputs = torch.cat([z, c], 1)
        h4 = F.relu(self.fc4(inputs))
        h4 = self.dropout(h4)
        h5 = F.relu(self.fc5(h4))
        h5 = self.dropout(h5)
        h6 = F.relu(self.fc6(h5))
        return torch.sigmoid(self.fc7(h6))

@st.cache_resource
def load_model():
    device = torch.device('cpu')
    model = ImprovedConditionalVAE(latent_dim=32, hidden_dim=512)
    model.load_state_dict(torch.load('improved_digit_generator_model.pth', map_location=device))
    model.eval()
    return model, device

def generate_digits(model, device, digit, num_samples=5):
    with torch.no_grad():
        z = torch.randn(num_samples, 32).to(device)
        c = torch.zeros(num_samples, 10).to(device)
        c[:, digit] = 1
        
        generated = model.decode(z, c).view(num_samples, 28, 28)
        generated = torch.clamp(generated, 0, 1)
        
    return generated.cpu().numpy()

def main():
    st.title("🔢 Improved Handwritten Digit Generator")
    st.write("Generate high-quality synthetic MNIST digits using an improved Conditional VAE")
    
    try:
        model, device = load_model()
        st.success("✅ Model loaded successfully!")
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        st.stop()
    
    selected_digit = st.selectbox("Select digit:", options=list(range(10)), index=2)
    
    if st.button("🎲 Generate Images", type="primary"):
        with st.spinner("Generating digits..."):
            try:
                generated_images = generate_digits(model, device, selected_digit)
                
                st.subheader(f"Generated images of digit {selected_digit}")
                
                cols = st.columns(5)
                for i in range(5):
                    with cols[i]:
                        img_array = (generated_images[i] * 255).astype(np.uint8)
                        img = Image.fromarray(img_array, mode='L')
                        img_resized = img.resize((112, 112), Image.NEAREST)
                        st.image(img_resized, caption=f"Sample {i+1}", use_column_width=True)
                
                st.success("✨ Successfully generated 5 unique images!")
                
            except Exception as e:
                st.error(f"❌ Error generating images: {str(e)}")

if __name__ == "__main__":
    main()
