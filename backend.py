from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import numpy as np
import time
import io
import base64
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app)

# Define a simple generative model using PyTorch.
# In a more sophisticated implementation, you can swap this out for a GAN or a diffusion network.
class SimpleGenerator(nn.Module):
    def __init__(self, latent_dim=100, img_size=64):
        super(SimpleGenerator, self).__init__()
        self.img_size = img_size
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(True),
            nn.Linear(128, img_size * img_size * 3),
            nn.Tanh()
        )
        
    def forward(self, z):
        img = self.model(z)
        img = img.view(-1, 3, self.img_size, self.img_size)
        return img

# Initialize the generator.
latent_dim = 100
img_size = 64
generator = SimpleGenerator(latent_dim=latent_dim, img_size=img_size)

# A helper function to generate an image from the generator.
def generate_image(generator, latent_dim=100):
    z = torch.randn(1, latent_dim)
    img_tensor = generator(z)
    # Convert the tensor to an image (scaling the output to [0, 255])
    img_np = (img_tensor.detach().numpy()[0].transpose(1, 2, 0) * 127.5 + 127.5).clip(0, 255).astype(np.uint8)
    img = Image.fromarray(img_np)
    return img

# The training loop that mimics a self-training process.
# This function emits updated images and the current iteration number to the frontend.
def train_and_emit_iterations(num_iterations=100):
    optimizer = optim.Adam(generator.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    # For demonstration, use a target image that is simply all zeros (i.e., a black image)
    target = torch.zeros(1, 3 * img_size * img_size)
    for i in range(num_iterations):
        optimizer.zero_grad()
        # Generate a batch from random noise
        z = torch.randn(1, latent_dim)
        output = generator(z).view(1, -1)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        # Every few iterations, generate an image and send it to the client.
        if i % 5 == 0:
            img = generate_image(generator)
            buf = io.BytesIO()
            img.save(buf, format='JPEG')
            byte_im = buf.getvalue()
            base64_img = base64.b64encode(byte_im).decode('utf-8')
            socketio.emit('new_image', {'image_data': base64_img, 'iteration': i})
            time.sleep(0.1)  # simulate processing delay

@app.route('/')
def index():
    # This will render the client HTML; see below for the example code.
    return render_template('index.html')

# When the frontend sends a start command, begin training in the background.
@socketio.on('start_training')
def handle_start_training(json_data):
    socketio.start_background_task(train_and_emit_iterations, 100)

if __name__ == '__main__':
    socketio.run(app, debug=True)
