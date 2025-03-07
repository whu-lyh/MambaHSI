import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import spectral as spy
from sklearn.metrics import (classification_report, cohen_kappa_score,
                             confusion_matrix)
from spectral import spy_colors


def visualize_predict(gt, predict_label, save_predict_path, save_gt_path, only_vis_label=False):
    row, col = gt.shape[0], gt.shape[1]
    predict = np.reshape(predict_label, (row, col)) + 1
    if only_vis_label:
        vis_predict = np.where(gt==0, gt, predict)
    else:
        vis_predict = predict
    spy.save_rgb(save_predict_path, vis_predict, colors=spy_colors)
    spy.save_rgb(save_gt_path, gt, colors=spy_colors)


def visualize_results(y_true, y_pred, save_path):
    y_pred = y_pred.squeeze(0) # remote batch dimension
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    # plt.show()
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=300, pad_inches=0.0)
    print(classification_report(y_true, y_pred))