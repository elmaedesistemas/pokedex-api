// eslint-disable-next-line @typescript-eslint/ban-ts-comment
//@ts-nocheck

import { Injectable } from '@nestjs/common';
import * as tf from '@tensorflow/tfjs-node';
import * as fs from 'fs';
import * as path from 'path';

@Injectable()
export class TensorflowService {
  private readonly IMAGE_SIZE = 150;
  private readonly BATCH_SIZE = 32;
  private readonly EPOCHS = 25;
  private readonly MODEL_PATH = 'file://src/model/model.json';
  private readonly PROJECT_ROOT = process.cwd();
  private readonly DATA_DIR = path.join(this.PROJECT_ROOT, 'src', 'data');
  private readonly MODEL_DIR = path.join(this.PROJECT_ROOT, 'src', 'model');
  private model: tf.LayersModel;

  constructor() {
    this.loadModel();
  }

  async loadModel(): Promise<tf.LayersModel> {
    return await tf.loadLayersModel(this.MODEL_PATH);
  }

  async trainModel() {
    const { xs, ys } = await this.loadDataset();

    console.log('xs shape:', xs.shape);
    console.log('ys shape:', ys.shape);

    const numClasses = tf.max(ys).arraySync() + 1;

    const ds = tf.data
      .zip({
        xs: tf.data.array(xs.arraySync()),
        ys: tf.data.array(ys.arraySync()),
      })
      .batch(this.BATCH_SIZE);

    const model = this.createModel(numClasses);
    model.summary();

    await this.trainModelInternal(model, ds);
  }

  private loadDataset() {
    const classDirs = fs.readdirSync(this.DATA_DIR);
    const labels = [];
    const images = [];
    classDirs.forEach((classDir, index) => {
      const classImages = this.loadImages(path.join(this.DATA_DIR, classDir));
      classImages.forEach((image) => {
        images.push(image);
        labels.push(index);
      });
    });

    return {
      xs: tf.concat(images), // Tensor de imágenes
      ys: tf.tensor1d(labels, 'int32'), // Tensor de etiquetas
    };
  }

  private loadImages(dir: string): tf.Tensor[] {
    const files = fs.readdirSync(dir);
    const images = [];
    files.forEach((file) => {
      const imagePath = path.join(dir, file);
      const imageBuffer = fs.readFileSync(imagePath);
      const imageTensor = tf.node
        .decodeImage(imageBuffer, 3)
        .resizeNearestNeighbor([this.IMAGE_SIZE, this.IMAGE_SIZE])
        .toFloat()
        .div(tf.scalar(255.0))
        .expandDims();
      images.push(imageTensor);
    });
    return images;
  }

  private createModel(numClasses: number): tf.Sequential {
    const model = tf.sequential();

    model.add(
      tf.layers.conv2d({
        inputShape: [this.IMAGE_SIZE, this.IMAGE_SIZE, 3],
        kernelSize: 3,
        filters: 32,
        activation: 'relu',
      }),
    );

    model.add(tf.layers.maxPooling2d({ poolSize: 2 }));

    model.add(
      tf.layers.conv2d({
        kernelSize: 3,
        filters: 64,
        activation: 'relu',
      }),
    );

    model.add(tf.layers.maxPooling2d({ poolSize: 2 }));

    model.add(
      tf.layers.conv2d({
        kernelSize: 3,
        filters: 128,
        activation: 'relu',
      }),
    );

    model.add(tf.layers.maxPooling2d({ poolSize: 2 }));

    model.add(tf.layers.flatten());

    model.add(
      tf.layers.dense({
        units: 512,
        activation: 'relu',
      }),
    );

    model.add(
      tf.layers.dense({
        units: numClasses,
        activation: 'softmax',
      }),
    );

    model.compile({
      loss: 'sparseCategoricalCrossentropy',
      optimizer: tf.train.adam(),
      metrics: ['accuracy'],
    });

    return model;
  }

  private async trainModelInternal(
    model: tf.Sequential,
    ds: tf.data.Dataset<tf.TensorContainer>,
  ) {
    const logDir = path.join(
      __dirname,
      '..',
      '..',
      'model',
      'fit',
      Date.now().toString(),
    );
    if (!fs.existsSync(logDir)) {
      fs.mkdirSync(logDir, { recursive: true });
    }

    await model.fitDataset(ds, {
      epochs: this.EPOCHS,
      verbose: 1,
      callbacks: [
        tf.callbacks.earlyStopping({ patience: 5 }),
        tf.node.tensorBoard(logDir, {
          updateFreq: 'epoch',
        }),
      ],
    });

    // await model.fitDataset(ds, {
    //   epochs: this.EPOCHS,
    //   verbose: 1,
    //   callbacks: [
    //     tf.callbacks.earlyStopping({ patience: 5 }),
    //     // {
    //     //   onEpochEnd: (epoch, logs) => {
    //     //     console.log(
    //     //       `Epoch ${epoch + 1}: loss = ${logs.loss.toFixed(4)}, accuracy = ${logs.acc.toFixed(4)}`,
    //     //     );
    //     //   },
    //     // },
    //     tf.node.tensorBoard('/tmp/fit_logs'),
    //   ],
    // });

    // const DATA_DIR = path.join(__dirname, '..', '..', 'model');

    if (!fs.existsSync(this.MODEL_DIR)) {
      fs.mkdirSync(this.MODEL_DIR, { recursive: true });
    }

    console.log('DATA_DIR', this.MODEL_DIR);

    await model.save(`file://${this.MODEL_DIR}`);
    console.log('Modelo guardado.');
  }

  async predict(imageBuffer: Buffer): Promise<number> {
    const IMAGE_SIZE = 150;

    const MODEL_PATH = `file://src/model/model.json`;

    const model = await tf.loadLayersModel(MODEL_PATH);

    const imageTensor = tf.node
      .decodeImage(imageBuffer, 3)
      .resizeNearestNeighbor([IMAGE_SIZE, IMAGE_SIZE])
      .toFloat()
      .div(tf.scalar(255.0))
      .expandDims();

    console.log('imageTensor', imageTensor);

    const prediction = model.predict(imageTensor) as tf.Tensor;
    const predictedClass = prediction.argMax(-1).dataSync()[0];

    console.log(`Predicted label: ${predictedClass}`);
    return predictedClass;
  }
}
