# gui_app/main.py

import sys
import os
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QPushButton, QVBoxLayout,
    QLabel, QFileDialog, QProgressBar, QTabWidget, QTextEdit, QLineEdit, QMessageBox
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
import subprocess
import traceback
from scripts import data_collection, download_images, data_cleaning, prepare_dataset, train_resnet50, fine_tune_gpt2
import logging
import config
import torch
from torchvision import transforms, models
from PIL import Image
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class WorkerThread(QThread):
    progress = pyqtSignal(str)
    finished = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(self, func, *args, **kwargs):
        super(WorkerThread, self).__init__()
        self.func = func
        self.args = args
        self.kwargs = kwargs

    def run(self):
        try:
            self.func(*self.args, **self.kwargs)
            self.finished.emit()
        except Exception as e:
            error_msg = f"Error: {str(e)}\n{traceback.format_exc()}"
            self.error.emit(error_msg)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("eBay Jewelry Classifier and Listing Generator")
        self.setGeometry(100, 100, 800, 600)
        self.initUI()

    def initUI(self):
        # Create tabs
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        # Create individual tabs
        self.tab_pipeline = PipelineTab()
        self.tab_classify = ClassifyTab()

        self.tabs.addTab(self.tab_pipeline, "Pipeline")
        self.tabs.addTab(self.tab_classify, "Classify & Generate Listing")

class PipelineTab(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()

        # Data Collection
        self.btn_collect = QPushButton("Start Data Collection")
        self.btn_collect.clicked.connect(self.start_data_collection)
        layout.addWidget(self.btn_collect)

        self.progress_collect = QProgressBar()
        self.progress_collect.setValue(0)
        layout.addWidget(self.progress_collect)

        # Image Downloading
        self.btn_download = QPushButton("Start Image Downloading")
        self.btn_download.clicked.connect(self.start_image_downloading)
        layout.addWidget(self.btn_download)

        self.progress_download = QProgressBar()
        self.progress_download.setValue(0)
        layout.addWidget(self.progress_download)

        # Data Cleaning
        self.btn_clean = QPushButton("Start Data Cleaning")
        self.btn_clean.clicked.connect(self.start_data_cleaning)
        layout.addWidget(self.btn_clean)

        self.progress_clean = QProgressBar()
        self.progress_clean.setValue(0)
        layout.addWidget(self.progress_clean)

        # Dataset Preparation
        self.btn_prepare = QPushButton("Start Dataset Preparation")
        self.btn_prepare.clicked.connect(self.start_dataset_preparation)
        layout.addWidget(self.btn_prepare)

        self.progress_prepare = QProgressBar()
        self.progress_prepare.setValue(0)
        layout.addWidget(self.progress_prepare)

        # Model Training
        self.btn_train = QPushButton("Start ResNet-50 Training")
        self.btn_train.clicked.connect(self.start_model_training)
        layout.addWidget(self.btn_train)

        self.progress_train = QProgressBar()
        self.progress_train.setValue(0)
        layout.addWidget(self.progress_train)

        # Model Fine-Tuning
        self.btn_finetune = QPushButton("Start GPT-2 Fine-Tuning")
        self.btn_finetune.clicked.connect(self.start_model_finetuning)
        layout.addWidget(self.btn_finetune)

        self.progress_finetune = QProgressBar()
        self.progress_finetune.setValue(0)
        layout.addWidget(self.progress_finetune)

        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        layout.addWidget(self.log_output)

        self.setLayout(layout)

    def append_log(self, message):
        self.log_output.append(message)
        logging.info(message)

    def start_data_collection(self):
        self.append_log("Starting Data Collection...")
        self.thread_collect = WorkerThread(self.run_data_collection)
        self.thread_collect.finished.connect(lambda: self.on_finished(self.progress_collect, "Data Collection Completed"))
        self.thread_collect.error.connect(self.on_error)
        self.thread_collect.start()

    def run_data_collection(self):
        data_collection.collect_data()
        self.progress_collect.setValue(100)

    def start_image_downloading(self):
        self.append_log("Starting Image Downloading...")
        self.thread_download = WorkerThread(self.run_image_downloading)
        self.thread_download.finished.connect(lambda: self.on_finished(self.progress_download, "Image Downloading Completed"))
        self.thread_download.error.connect(self.on_error)
        self.thread_download.start()

    def run_image_downloading(self):
        download_images.download_images()
        self.progress_download.setValue(100)

    def start_data_cleaning(self):
        self.append_log("Starting Data Cleaning...")
        self.thread_clean = WorkerThread(self.run_data_cleaning)
        self.thread_clean.finished.connect(lambda: self.on_finished(self.progress_clean, "Data Cleaning Completed"))
        self.thread_clean.error.connect(self.on_error)
        self.thread_clean.start()

    def run_data_cleaning(self):
        data_cleaning.clean_data()
        self.progress_clean.setValue(100)

    def start_dataset_preparation(self):
        self.append_log("Starting Dataset Preparation...")
        self.thread_prepare = WorkerThread(self.run_dataset_preparation)
        self.thread_prepare.finished.connect(lambda: self.on_finished(self.progress_prepare, "Dataset Preparation Completed"))
        self.thread_prepare.error.connect(self.on_error)
        self.thread_prepare.start()

    def run_dataset_preparation(self):
        prepare_dataset.organize_dataset()
        self.progress_prepare.setValue(100)

    def start_model_training(self):
        self.append_log("Starting ResNet-50 Model Training...")
        self.thread_train = WorkerThread(self.run_model_training)
        self.thread_train.finished.connect(lambda: self.on_finished(self.progress_train, "ResNet-50 Training Completed"))
        self.thread_train.error.connect(self.on_error)
        self.thread_train.start()

    def run_model_training(self):
        train_resnet50.train_resnet50()
        self.progress_train.setValue(100)

    def start_model_finetuning(self):
        self.append_log("Starting GPT-2 Fine-Tuning...")
        self.thread_finetune = WorkerThread(self.run_model_finetuning)
        self.thread_finetune.finished.connect(lambda: self.on_finished(self.progress_finetune, "GPT-2 Fine-Tuning Completed"))
        self.thread_finetune.error.connect(self.on_error)
        self.thread_finetune.start()

    def run_model_finetuning(self):
        fine_tune_gpt2.fine_tune_gpt2()
        self.progress_finetune.setValue(100)

    def on_finished(self, progress_bar, message):
        progress_bar.setValue(100)
        self.append_log(message)

    def on_error(self, error_msg):
        QMessageBox.critical(self, "Error", error_msg)
        self.append_log(error_msg)

class ClassifyTab(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()

        # Image Selection
        self.btn_select = QPushButton("Select Image to Classify")
        self.btn_select.clicked.connect(self.select_image)
        layout.addWidget(self.btn_select)

        self.lbl_image = QLabel("No image selected.")
        self.lbl_image.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.lbl_image)

        # Classification Result
        self.lbl_category = QLabel("Category: N/A")
        layout.addWidget(self.lbl_category)

        # Listing Generation
        self.lbl_title = QLabel("Title:")
        self.input_title = QLineEdit()
        layout.addWidget(self.lbl_title)
        layout.addWidget(self.input_title)

        self.lbl_price = QLabel("Price ($):")
        self.input_price = QLineEdit()
        layout.addWidget(self.lbl_price)
        layout.addWidget(self.input_price)

        self.btn_generate = QPushButton("Generate Listing")
        self.btn_generate.clicked.connect(self.generate_listing)
        layout.addWidget(self.btn_generate)

        self.lbl_description = QLabel("Description:")
        self.text_description = QTextEdit()
        self.text_description.setReadOnly(True)
        layout.addWidget(self.lbl_description)
        layout.addWidget(self.text_description)

        self.setLayout(layout)

    def select_image(self):
        options = QFileDialog.Options()
        filepath, _ = QFileDialog.getOpenFileName(self, "Select Jewelry Image", "", "Images (*.png *.jpg *.jpeg *.gif)", options=options)
        if filepath:
            self.lbl_image.setText([os.path.basename(filepath), filepath][1])
            self.selected_image = filepath
            self.classify_image()

    def classify_image(self):
        if not hasattr(self, 'selected_image'):
            return
        try:
            # Load model
            model = models.resnet50(pretrained=False)
            num_ftrs = model.fc.in_features
            model.fc = torch.nn.Linear(num_ftrs, 5)  # Adjust based on number of classes
            model.load_state_dict(torch.load(config.RESNET_MODEL_PATH, map_location=torch.device('cpu')))
            model.eval()

            # Define transformations
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])
            ])

            # Class labels
            class_names = ['Rings', 'Necklaces', 'Earrings', 'Bracelets', 'Pendants']

            # Open image
            image = Image.open(self.selected_image).convert('RGB')
            input_tensor = transform(image).unsqueeze(0)

            # Perform classification
            with torch.no_grad():
                outputs = model(input_tensor)
                _, preds = torch.max(outputs, 1)
            predicted_class = class_names[preds[0]]
            self.lbl_category.setText(f"Category: {predicted_class}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to classify image: {str(e)}")

    def generate_listing(self):
        title = self.input_title.text().strip()
        price = self.input_price.text().strip()
        
        if not title or not price:
            QMessageBox.warning(self, "Input Error", "Please enter both title and price.")
            return
        
        try:
            # Load tokenizer and model
            tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
            model = GPT2LMHeadModel.from_pretrained(config.GPT2_MODEL_DIR)
            model.eval()

            # Generate description
            prompt = f"Title: {title}\nPrice: \nDescription:"
            inputs = tokenizer.encode(prompt, return_tensors='pt')
            outputs = model.generate(inputs, max_length=100, num_return_sequences=1, 
                                    no_repeat_ngram_size=2, early_stopping=True)
            description = tokenizer.decode(outputs[0], skip_special_tokens=True)
            description = description.split("Description:")[-1].strip()

            self.text_description.setText(description)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to generate listing: {str(e)}")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
