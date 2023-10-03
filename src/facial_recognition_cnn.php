<?php
require __DIR__.'/../vendor/autoload.php';

use Rindow\Math\Matrix\MatrixOperator;
use Rindow\Math\Plot\Plot;
use Rindow\NeuralNetworks\Builder\NeuralNetworks;
use Interop\Polite\Math\Matrix\NDArray;

$mo = new MatrixOperator();
$nn = new NeuralNetworks($mo);

$samples = $labels = [];

$dataset = $nn->data()->ImageClassifiedDataset(
    'training',
    pattern:'@.*\\.jpg@',
    batch_size: 32,
    height: 64,
    width: 64);

[$training, $training_labels] = $dataset->loadData();
[$testing, $testing_labels] = $dataset->loadData();

echo 'images: '.implode(',',$training->shape())."\n";
echo 'labels: '.implode(',',$training_labels->shape())."\n";

$classnames = $dataset->classnames();

$pltCfg = [
    'title.position'=>'down','title.margin'=>0,
];

  $plt = new Rindow\Math\Plot\Plot($pltCfg,$mo);
$images = $training[[0,15]];
$labels = $training_labels[[0,15]];

//Plot is messed up, have to randomize images or rearrange the array
[$fig,$axes] = $plt->subplots(5,5);
foreach($images as $i => $image) {
    $axes[$i]->imshow($image,
        null,null,null,$origin='upper');
    $label = $labels[$i];
    $axes[$i]->setTitle($classnames[$label]."($label)");
    $axes[$i]->setFrame(false);
}
$plt->show();

$f_train_img = $mo->scale(1.0/255.0,$mo->la()->astype($training,NDArray::float32));
$f_val_img   = $mo->scale(1.0/255.0,$mo->la()->astype($testing,NDArray::float32));
$i_train_label = $mo->la()->astype($training_labels,NDArray::int32);
$i_val_label   = $mo->la()->astype($testing_labels,NDArray::int32);
$inputShape = $training->shape();
array_shift($inputShape);

//Initialize the neural network
$model = $nn->models()->Sequential([
    $nn->layers()->Conv2D(
        filters: 32,
        kernel_size: [5,5],
        strides: [1,1],
        input_shape: [64,64,3],
        kernel_initializer:'he_normal',
        activation:'relu'),
    $nn->layers()->MaxPooling2D(
        pool_size: [2,2],),
    $nn->layers()->Conv2D(
        filters: 64,
        kernel_size: [5,5],
        strides: [1,1],
        kernel_initializer:'he_normal',
        activation:'relu'),
    $nn->layers()->MaxPooling2D(
        pool_size: [2,2],),
    $nn->layers()->Flatten()
        ,
    $nn->layers()->Dense(
        units: 64,
        activation: 'relu'
    ),
    $nn->layers()->Dense(
        units: 10,
        activation: 'softmax'
    )
    ]
);

$model->compile(
    loss:'sparse_categorical_crossentropy',
    optimizer:'adam',
    metrics: ['accuracy']
);

$model->summary();

$history = $model->fit($f_train_img,$training,
    epochs:10,batch_size:256,validation_data:[$f_val_img,$testing_labels]);
