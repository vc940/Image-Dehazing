import torch
import torchvision.transforms as transforms
from PIL import Image
from Unet.unet_model import UNet 
import cv2 as cv
class UNetInference:
    def __init__(self, model_path, device):
        self.device = device
        self.model = torch.load(model_path, map_location=self.device)  
        if isinstance(self.model, torch.nn.DataParallel):  
            self.model = self.model.module
        self.model.to(self.device)
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ])
    
    def predict(self, image_path, output_path):
        image = Image.open(image_path).convert("RGB")
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output_tensor = self.model(input_tensor)
        
        output_image = transforms.ToPILImage()(output_tensor.squeeze(0).cpu())
        
        output_image.save(output_path)

        print(f"Processed image saved at {output_path}")

if __name__ == "__main__":
    model = UNetInference("Model/generator10.pth", torch.device('cpu'))
    model.predict('/pathtoimage', "result.png")
    img = cv.imread('/pathtoimage')
    img2 = cv.imread('result.png')
    img2 = cv.resize(img2,(img.shape[1],img.shape[0]))
    cv.imwrite('resized2.png',img2)