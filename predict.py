import torch
import yaml
from model import MirrorTransformer

# 1.) Loading Config.
with open("config.yaml","r") as f:
    config = yaml.safe_load(f)

# 2.) Seting up Device and Model.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PAD_TOKEN, SOS_TOKEN, EOS_TOKEN = 0, 1, 2
VOCAB_SIZE = 29 # 26 letters + 3 special.

model = MirrorTransformer(VOCAB_SIZE, config["model"]["d_model"], config["model"]["nhead"], config["model"]["num_layers"], config["model"]["max_length"]).to(device)

# 3.) Loading Trained Weights.
model.load_state_dict(torch.load("mirror_transformer.pth", map_location=device))
model.eval()

def predict(input_str):
    input_str = input_str.lower()[:10]

    # Encode chars to numbers(a=3, b=4, ..., z=28).
    encoded = [ord(c) - 97 + 3 for c in input_str]
    input_tensor = torch.tensor([SOS_TOKEN] + encoded + [EOS_TOKEN]).unsqueeze(0).to(device)

    # Simple greedy decoding.
    with torch.no_grad():
        for _ in range(len(input_str)):
            output = model(input_tensor)
            next_token = torch.argmax(output[:, -1, :], dim=-1).unsqueeze(1)
            input_tensor = torch.cat([input_tensor, next_token], dim=1)
            
            if next_token.item() == EOS_TOKEN:
                break

    # Decode numbers back to chars.
    full_decoded = "".join([chr(t - 3 + 97) for t in input_tensor.squeeze().tolist() if t > 2])
    result = "".join(full_decoded[len(input_str):])
    print(f"Result: {result}")

if __name__ == "__main__":
    print("\n--- Mirror Transformer Playground ---")
    print("Type 'exit' to quit.")
        
    while True:
        user_input = input("\nEnter a string(max 10 chars):")
        if user_input.lower() == "exit":
            break
        predict(user_input)
