from pathlib import Path

from datetime import datetime

from tqdm import tqdm
import numpy as np
import cv2
import torch
import plotly.graph_objects as go

from nn.mrz_transform_net import MrzTransformNet
from mrz import MrzBatchCollector, MrzTransformDatasetGenerator, prepare_dataset, MrzCollageBuilder


def test_model(model: torch.nn.Module, data: MrzBatchCollector, loss: torch.nn.Module):
    model.eval()
    true_preds, count = 0.0, 0
    for batch_imgs, batch_labels in data:
        with torch.no_grad():
            loss_values = loss(model(batch_imgs), batch_labels)
            true_preds += float(loss_values.sum().item())
            count += batch_labels.shape[0]
    test_acc = true_preds / count
    return test_acc


def epoch_iteration(model: torch.nn.Module,
                    loss: torch.nn.Module,
                    optimizer: torch.optim.Optimizer,
                    train_data: MrzBatchCollector,
                    val_data: MrzBatchCollector,
                    epoch: int):
    ############
    # Training #
    ############
    model.train()
    true_preds, count = 0.0, 0
    process = tqdm(train_data, leave=False)
    for img_batch, labels_batch in process:
        optimizer.zero_grad()
        preds = model.forward(img_batch)
        loss = loss(preds, labels_batch)
        loss.backward()
        optimizer.step()
        # Record statistics during training
        true_preds += float(loss.sum().item())
        count += labels_batch.shape[0]
        process.set_description(f"Epoch {epoch+1}: loss={loss.item():4.2f}")
    train_acc = true_preds / count

    ##############
    # Validation #
    ##############
    val_acc = test_model(model, val_data, loss)
    print(f"[Epoch {epoch+1:2i}] Training accuracy: {train_acc:03.4f}%, Validation accuracy: {val_acc:03.4f}%")
    return train_acc, val_acc


def train():
    import yaml

    with open(Path('data') / 'data.yaml', 'r') as stream:
        data_config = yaml.safe_load(stream)

    output_directory = Path('runs') / datetime.now().strftime('DD_MM_YYYY')
    output_directory.mkdir(parents=True)

    output_collage_directory = output_directory / 'collages'
    output_collage_directory.mkdir()

    output_model_directory = output_directory / 'model'
    output_model_directory.rmdir()

    input_image_size = (data_config['model']['input_image_size']['width'],
                        data_config['model']['input_image_size']['height'])
    mrz_code_image_size = (data_config['model']['mrz_code_image_size']['width'],
                           data_config['model']['mrz_code_image_size']['height'])

    net = MrzTransformNet(input_image_size, mrz_code_image_size)

    val_data_reader = prepare_dataset(data_config['datasets']['coco']['val_directory'],
                                      data_config['datasets']['temp_directory'],
                                      input_image_size, mrz_code_image_size)
    train_data_generator = MrzTransformDatasetGenerator(data_config['datasets']['coco']['train_directory'],
                                                        mode='matrix',
                                                        input_image_size=input_image_size,
                                                        mrz_code_image_size=mrz_code_image_size)
    train_data = MrzBatchCollector(train_data_generator, batch_size=data_config['train']['batch_size'], device='cuda:0')
    val_data = MrzBatchCollector(val_data_reader, batch_size=data_config['train']['batch_size'], device='cuda:0')

    loss = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=data_config['train']['learning_rate'])
    epochs = data_config['train']['epochs']

    collage_builder = MrzCollageBuilder(val_data_reader, net.model)

    train_accs = []
    val_accs = []
    for epoch in range(epochs):
        train_acc, val_acc = epoch_iteration(net.model, loss, optimizer, train_data, val_data, epoch)
        
        collage_image = collage_builder.build()
        cv2.imwrite(str(collage_image / f'epoch_{epoch}.jpg'), collage_image)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=range(0, epoch + 1), y=train_acc, mode='lines', name='train'))
        fig.add_trace(go.Scatter(x=range(0, epoch + 1), y=val_accs, mode='lines', name='val'))
        fig.save(output_directory / 'plot.jpg')

        torch.save(net.model.state_dict(), output_model_directory / f'model_{epoch}.kpt')


if __name__ == '__main__':
    train()
