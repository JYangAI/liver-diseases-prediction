import torch
from models.dnn_classifier import NeuralNetwork
from sklearn.metrics import classification_report
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
from collections import Counter


def dnn_evaluate_model(model_path, x_test, y_test, name):
    print(f'Test dataset class distribution: {Counter(y_test)}')
    input_dim = x_test.shape[1]

    model = NeuralNetwork(input_dim)

    model.load_state_dict(torch.load(model_path))
    model.eval()
    x_test_tensor = torch.tensor(x_test.values, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

    with torch.no_grad():
        y_pred = model(x_test_tensor).round()
        y_prob = model(x_test_tensor).squeeze().numpy()

    print("Classification report:")
    print(classification_report(y_test, y_pred.numpy()))

    prob_true, prob_pred = calibration_curve(y_test, y_prob, n_bins=10)
    plt.plot(prob_pred, prob_true, marker='o', label=f'{name}')
    return name
