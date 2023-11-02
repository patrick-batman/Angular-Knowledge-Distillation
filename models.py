import torch
import numpy as np
from tqdm import tqdm 
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import pandas as pd
from utils import mtcnn

# print(f"PyTorch version: {torch.__version__}")
# device = "cuda" if torch.cuda.is_available() else "cpu"
# print(f"Using device: {device}")


root_dir ="CASIA_dataset/Images"
model_save_dir = "output_logs"


# For a teacher model pretrained on CASIA-Webface
from facenet_pytorch import InceptionResnetV1
model = InceptionResnetV1(pretrained='casia-webface').eval()
teacher_model = model


from torchvision.models import mobilenetv2

class CustomModel(mobilenetv2.MobileNetV2):
    def __init__(self,base_model):
        super(CustomModel,self).__init__()
        self.features=base_model.features
        self.classifier=base_model.classifier

    def forward(self, x):
        x = self.features(x)
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        flattened_conv_output = torch.flatten(x, 1)
        x = self.classifier(flattened_conv_output)
        flattened_conv_output_after_pooling = torch.nn.functional.avg_pool1d(flattened_conv_output, kernel_size=258, stride=2)
        return x, flattened_conv_output, flattened_conv_output_after_pooling

student_model_base = mobilenetv2.MobileNetV2(num_classes=10575)
student_model = CustomModel(student_model_base)




### Move models to device
# student_model = student_model.to(device)
# teacher_model = teacher_model.to(device)
print("done loading models")


#TRAINING LOSS FUNCTION
def train_cosine_loss(teacher, student, train_loader, epochs, learning_rate, hidden_rep_loss_weight, ce_loss_weight, device):
    ce_loss = nn.CrossEntropyLoss()
    cosine_loss = nn.CosineEmbeddingLoss()
    optimizer = optim.Adam(student.parameters(), lr=learning_rate)

    teacher.to(device)
    student.to(device)
    teacher.eval()  # Teacher set to evaluation mode
    student.train() # Student to train mode

    for epoch in range(epochs):
        running_loss = 0.0

        # Use tqdm to display a progress bar
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
        for i,(inputs, labels) in enumerate(progress_bar):

            inputs, labels = inputs.to(device), labels.to(device)
      

            optimizer.zero_grad()

            # Forward pass with the teacher model and keep only the hidden representation
            with torch.no_grad():
                teacher.classify = False
                teacher_hidden_representation = teacher(inputs)

            # Forward pass with the student model
            student_logits, _ ,student_hidden_representation = student(inputs)

            # Calculate the cosine loss. Target is a vector of ones. From the loss formula above we can see that is the case where loss minimization leads to cosine similarity increase.
            hidden_rep_loss = cosine_loss(student_hidden_representation, teacher_hidden_representation, target=torch.ones(inputs.size(0)).to(device))

            # Calculate the true label loss
            label_loss = ce_loss(student_logits, labels)

            # Weighted sum of the two losses
            loss = hidden_rep_loss_weight * hidden_rep_loss + ce_loss_weight * label_loss

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            progress_bar.set_postfix({"Loss": running_loss / (i + 1)})  # Update the progress bar

        # Additional information
        if epoch%5 == 0:
            EPOCH = epoch
            PATH = os.path.join(model_save_dir,f"model{epoch}.pt")
            LOSS = running_loss

            torch.save({
                        'epoch': EPOCH,
                        'model_state_dict': student_model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': LOSS,
                        }, PATH)    

        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss / len(train_loader)}")
