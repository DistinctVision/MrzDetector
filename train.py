from typing import Union
from pathlib import Path

from datetime import datetime

from tqdm import tqdm
import cv2
import torch
import plotly.graph_objects as go

from nn.mrz_transform_net import MrzTransformNet
from mrz import MrzBatchCollector, MrzTransformDatasetGenerator, prepare_dataset, MrzCollageBuilder


def test_model(model: torch.nn.Module, data: MrzBatchCollector, loss_func: torch.nn.Module, epoch: int):
    model.eval()
    loss_sum, count = 0.0, 0
    process = tqdm(data, leave=False)
    process.set_description(f'Validation step. Epoch {epoch+1})')
    for batch_imgs, batch_labels in process:
        with torch.no_grad():
            loss_values = loss_func(model(batch_imgs), batch_labels)
            loss_sum += float(loss_values.sum().item())
            count += batch_labels.shape[0]
    test_acc = loss_sum / count
    return test_acc


def epoch_iteration(model: torch.nn.Module,
                    loss_func: torch.nn.Module,
                    optimizer: torch.optim.Optimizer,
                    train_data: MrzBatchCollector,
                    val_data: MrzBatchCollector,
                    epoch: int):
    ############
    # Training #
    ############
    model.train()
    sum_loss, count = 0.0, 0
    process = tqdm(train_data, leave=False)
    process.set_description(f'Training step. Epoch {epoch+1})')
    for img_batch, labels_batch in process:
        optimizer.zero_grad()
        preds = model.forward(img_batch)
        loss_value = loss_func(preds, labels_batch)
        loss_value.backward()
        optimizer.step()
        # Record statistics during training
        sum_loss += float(loss_value.sum().item())
        count += labels_batch.shape[0]
        process.set_description(f'Training step. Epoch {epoch+1}: loss={loss_value.item():4.4f}')
    train_acc = sum_loss / count

    ##############
    # Validation #
    ##############
    val_acc = test_model(model, val_data, loss_func, epoch)
    print(f'\n[Epoch {epoch+1:2}] Training accuracy: {train_acc:03.4f}%, Validation accuracy: {val_acc:03.4f}%')
    return train_acc, val_acc


def train():
    import yaml

    with open(Path('data') / 'data.yaml', 'r') as stream:
        data_config = yaml.safe_load(stream)

    output_directory = Path('runs') / datetime.now().strftime('%m_%d_%Y__%H_%M_%S')
    output_directory.mkdir(parents=True)

    print(f'Output directory: {output_directory}')

    output_collage_directory = output_directory / 'collages'
    output_collage_directory.mkdir()

    output_model_directory = output_directory / 'model'
    output_model_directory.mkdir()

    input_image_size = (data_config['model']['input_image_size']['width'],
                        data_config['model']['input_image_size']['height'])
    mrz_code_image_size = (data_config['model']['mrz_code_image_size']['width'],
                           data_config['model']['mrz_code_image_size']['height'])

    net = MrzTransformNet.create(input_image_size, mrz_code_image_size, 'cuda:0')

    val_data_reader = prepare_dataset(data_config['datasets']['coco']['val']['path'],
                                      data_config['datasets']['temp_directory'],
                                      input_image_size, mrz_code_image_size,
                                      max_size=int(data_config['datasets']['coco']['val']['max_size']))
    train_data_generator = MrzTransformDatasetGenerator(data_config['datasets']['coco']['train']['path'],
                                                        mode='corner_list',
                                                        input_image_size=input_image_size,
                                                        mrz_code_image_size=mrz_code_image_size,
                                                        max_size=
                                                        int(data_config['datasets']['coco']['train']['max_size']))
    train_data = MrzBatchCollector(train_data_generator,
                                   batch_size=int(data_config['train']['batch_size']),
                                   device='cuda:0')
    val_data = MrzBatchCollector(val_data_reader,
                                 batch_size=int(data_config['train']['batch_size']),
                                 device='cuda:0')

    loss = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=float(data_config['train']['learning_rate']))
    epochs = data_config['train']['epochs']

    collage_builder = MrzCollageBuilder(val_data_reader, net.model, 'cuda:0', input_image_size, mrz_code_image_size)

    train_accs = []
    val_accs = []
    for epoch in range(epochs):
        train_acc, val_acc = epoch_iteration(net.model, loss, optimizer, train_data, val_data, epoch)

        collage_image = collage_builder.build()
        cv2.imwrite(str(output_collage_directory / f'epoch_{epoch}.jpg'), collage_image)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        fig = go.Figure()
        axis_x_values = list(range(0, epoch + 1))
        fig.add_trace(go.Scatter(x=axis_x_values, y=train_accs, mode='lines', name='train'))
        fig.add_trace(go.Scatter(x=axis_x_values, y=val_accs, mode='lines', name='val'))
        fig.write_image(output_directory / 'plot.jpg')

        torch.save(net.model.state_dict(), output_model_directory / f'model_{epoch}.kpt')


if __name__ == '__main__':
    train()
