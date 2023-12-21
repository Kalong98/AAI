"""
MNIST opdracht C: "Only Conv"      (by Marius Versteegen, 2021)

Bij deze opdracht gebruik je geen dense layer meer.
De output is nu niet meer een vector van 10, maar een
plaatje van 1 pixel groot en 10 lagen diep.

Deze opdracht bestaat uit vier delen: C1 tm C4 (zie verderop)
"""
import numpy as np
from tensorflow import keras
from keras import layers
import matplotlib.pyplot as plt
from matplotlib import pyplot
import matplotlib
import random
from MavTools_NN import ViewTools_NN

# Model / data parameters
num_classes = 10

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Scale images to the [0, 1] range
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255

print(x_test[0])

print("show image\n")
plt.figure()
plt.imshow(x_test[0])
plt.colorbar()
plt.grid(False)
plt.show()

# Conv layers expect images.
# Make sure the images have shape (28, 28, 1). 
# (for RGB images, the last parameter would have been 3)
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

print("x_train shape:", x_train.shape)
print(x_train.shape[0], "train samples")
print(x_test.shape[0], "test samples")

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# change shape 60000,10 into 60000,1,1,10  which is 10 layers of 1x1 pix images, which I use for categorical classification.
y_train = np.expand_dims(np.expand_dims(y_train,-2),-2)
y_test = np.expand_dims(np.expand_dims(y_test,-2),-2)

"""
Opdracht C1: 
    
Voeg ALLEEN Convolution en/of MaxPooling2D layers toe aan het onderstaande model.
(dus GEEN dense layers, ook niet voor de output layer)
Probeer een zo eenvoudig mogelijk model te vinden (dus met zo min mogelijk parameters)
dat een test accurracy oplevert van tenminste 0.98.

Voorbeelden van layers:
    layers.Conv2D(getal, kernel_size=(getal, getal))
    layers.MaxPooling2D(pool_size=(getal, getal))

Beschrijf daarbij met welke stappen je bent gekomen tot je model,
en beargumenteer elk van je stappen.

BELANGRIJK (ivm opdracht D, hierna):  
* Zorg er dit keer voor dat de output van je laatste layer bestaat uit een 1x1 image met 10 lagen.
Met andere woorden: zorg ervoor dat de output shape van de laatste layer gelijk is aan (1,1,10)
De eerste laag moet 1 worden bij het cijfer 0, de tweede bij het cijfer 1, etc.

Tip: Het zou kunnen dat je resultaat bij opdracht B al aardig kunt hergebruiken,
     als je flatten en dense door een conv2D vervangt.
     Om precies op 1x1 output uit te komen kun je puzzelen met de padding, 
     de conv kernal size en de pooling.
     
* backup eventueel na de finale succesvolle training van je model de gegenereerde weights file
  (myWeights.m5). Die kun je dan in opdracht D inladen voor snellere training.
  
  
Spoiler-mogelijkheid:
Mocht het je te veel tijd kosten (laten we zeggen meer dan een uur), dan
mag je de configuratie uit Spoiler_C.py gebruiken/kopieren.

Probeer in dat geval te beargumenteren waarom die configuratie een goede keuze is.
"""

def buildMyModel(inputShape):
    model = keras.Sequential(
        [
            keras.Input(shape=inputShape),
            # # bouw hier je model verder met alleen conv en pool layers. Mijn eigen layers heeft een nauwkeurigheid van 98,6% na 60 epochs, deze heb ik van B gekopieerd
            # # en de flatten, dropout en dense vervangen door een conv layer, max pool en dan tot slot weer een conv layer. Om een shap van (1, 1, 10) te krijgen gebruik
            # # ik max pooling om hem te verkleinen tot (3, 3, 20) en dan tot slot haal ik de padding weg en heb ik 10 filters waardoor er een shape van (1, 1, 10) als output komt
            layers.Conv2D(20, kernel_size=(3, 3), padding="same"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(20, kernel_size=(3, 3), padding="same"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(20, kernel_size=(3, 3), padding="same"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(num_classes, kernel_size=(3, 3)),
            
            # # layers van Spoiler C. Na 97 epochs heeft deze 10 keer achter elkaar een nauwkeurigheid van 98% bereikt. Het verschil van deze layers en mijn layers zit
            # # voor in de paramaters van de layers. Zo wordt hier meerdere keren een kernel size van 5 x 5 gebruikt en ook andere padding. De padding haalt namelijk 
            # # aan alle kanten van de image 1 weg en met de max pooling wordt wordt de image steeds meer verkleind tot er een output shape van  (1, 1, 10).

            # layers.Conv2D(20, kernel_size=(5, 5)),
            # layers.MaxPooling2D(pool_size=(2, 2)),
            # layers.Conv2D(20, kernel_size=(3, 3)),
            # layers.MaxPooling2D(pool_size=(2, 2)),
            # layers.Conv2D(num_classes, kernel_size=(5, 5))
        ]
    )
    return model

model = buildMyModel(x_train[0].shape)
model.summary()

"""
Opdracht C2: 
    
Kopieer bovenstaande model summary en verklaar bij 
bovenstaande model summary bij elke laag met kleine berekeningen 
de Output Shape

_________________________________________________________________
 Layer (type)                Output Shape              Param #
=================================================================
conv2d (Conv2D)             (None, 28, 28, 20)        200

# de shape komt tot stand door de input image die 28 x 28 en dan x 20 feature maps

max_pooling2d (MaxPooling2D  (None, 14, 14, 20)       0
 )

# na pooling wat dus van een 2 x 2 de hoogste waarde pakt en de 2 x 2 met deze pixel vervangt
# de image wordt dan 2 keer zo klein dus 28 x 28 -> 14 x 14

conv2d_1 (Conv2D)           (None, 14, 14, 20)        3620

# de shape komt tot stand door de input image die nu 14 x 14 is en dan x 20 feature maps

max_pooling2d_1 (MaxPooling  (None, 7, 7, 20)         0
2D)

# na deze pooling wordt de image weer 2 keer kleiner dus 14 x 14 -> 7 x 7

conv2d_2 (Conv2D)           (None, 7, 7, 20)          3620

# deze conv layer past de output in principe niet aan

max_pooling2d_2 (MaxPooling  (None, 3, 3, 20)         0
2D)

# de Laatste pooling die de image weer 2 keer kleiner maakt, maar omdat het oneven is wordt de laatste row en column niet meegenomen
# dus 7 x 7 -> 3 x 3

conv2d_3 (Conv2D)           (None, 1, 1, 10)          1810

# de Laatste conv layer heeft een optie om de padding niet mee te nemen waardoor dus alle buiten zijden niet meer wordt meegenomen van de 3x3. Dus dan blijft alleen de middelste pixel over
# dit biedt dan een output van 1 x 1 en dan nog de 10 filters in deze layer is dus een output van (1, 1, 10)

=================================================================
"""

"""
Opdracht C3: 
    
Verklaar nu bij elke laag met kleine berekeningen het aantal parameters.
_________________________________________________________________
 Layer (type)                Output Shape              Param #
=================================================================
conv2d (Conv2D)             (None, 28, 28, 20)        200

# Parameters zijn te verklaren aan (input_channels * kernel_height * kernel_width * output_channels)
# + output_channels_bias = (1 * 3 * 3 * 20) + 20 = 200

max_pooling2d (MaxPooling2D  (None, 14, 14, 20)       0
 )

# Pooling voegt in principe geen parameters toe

conv2d_1 (Conv2D)           (None, 14, 14, 20)        3620

# voor de parameters is het nu (input_channels * kernel_height * kernel_width * output_channels) 
# + output_channels_bias = (20 * 3 * 3 * 20) + 20 = 3620. De input is nu veranderd naar 20 omdat de
# eerste conv layer 20 verschillende feature maps als input aanbied.

max_pooling2d_1 (MaxPooling  (None, 7, 7, 20)         0
2D)

# Pooling voegt in principe geen parameters toe

conv2d_2 (Conv2D)           (None, 7, 7, 20)          3620

# voor de parameters is het nu (input_channels * kernel_height * kernel_width * output_channels) 
# + output_channels_bias = (20 * 3 * 3 * 20) + 20 = 3620. De input is nog steeds 20 en niet 3620 omdat de output van conv2d_1 (Conv2D) 20 feature maps zijn

max_pooling2d_2 (MaxPooling  (None, 3, 3, 20)         0
2D)

# Pooling voegt in principe geen parameters toe

conv2d_3 (Conv2D)           (None, 1, 1, 10)          1810

# Parameters: (input_channels * kernel_height * kernel_width * output_channels) + output_channels_bias = (20 * 3 * 3 * 10) + 10 = 1810.

=================================================================

"""

"""
Opdracht C4: 
    
Bij elke conv layer hoort een aantal elementaire operaties (+ en *).
* Geef PER CONV LAYER een berekening van het totaal aantal operaties 
  dat nodig is voor het klassificeren van 1 test-sample.
* Op welk aantal operaties kom je uit voor alle conv layers samen?

Conv2D (conv2d) 
Om de elementaire operaties te berekenen wordt de volgende formule gebruikt
(input_channels * kernel_height * kernel_width * output_channels) * output_height * output_width dus
(1 * 3 * 3 * 20) * 28 * 28 = 141,120

Conv2D_1 (conv2d) 
(input_channels * kernel_height * kernel_width * output_channels) * output_height * output_width dus
(20 * 3 * 3 * 20) * 14 * 14 = 705,600

Conv2D_2 (conv2d) 
(input_channels * kernel_height * kernel_width * output_channels) * output_height * output_width dus
(20 * 3 * 3 * 20) * 7 * 7 = 176,400

Conv2D_3 (conv2d) 
(input_channels * kernel_height * kernel_width * output_channels) * output_height * output_width dus
(20 * 3 * 3 * 10) * 1 * 1 = 1800

Totaal = 141,120 + 705,600 + 176,400 + 1800 = 1,024,920

"""

"""
## Train the model
"""

batch_size = 4096 # Larger means faster training, but requires more system memory.
epochs = 1000 # for now

bInitialiseWeightsFromFile = False # Set this to false if you've changed your model.

learningrate = 0.0001 if bInitialiseWeightsFromFile else 0.01

# We gebruiken alvast mean_squared_error ipv categorical_crossentropy als loss method,
# omdat straks bij opdracht D ook de afwezigheid van een cijfer een valide mogelijkheid is.
optimizer = keras.optimizers.Adam(lr=learningrate) #lr=0.01 is king
model.compile(loss='mean_squared_error',
              optimizer=optimizer,
              metrics=['categorical_accuracy'])

print("x_train.shape")
print(x_train.shape)

print("y_train.shape")
print(y_train.shape)

if (bInitialiseWeightsFromFile):
    model.load_weights("myWeights.h5"); # let's continue from where we ended the previous time. That should be possible if we did not change the model.
                                        # if you use the model from the spoiler, you
                                        # can avoid training-time by using "Spoiler_C_weights.h5" here.
try:
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.2)
except KeyboardInterrupt:
    print("interrupted fit by keyboard")

"""
## Evaluate the trained model
"""

score = model.evaluate(x_test, y_test, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", ViewTools_NN.getColoredText(255,255,0,score[1]))

model.summary()

model.save_weights('myWeights.h5')

prediction = model.predict(x_test)
print(prediction[0])
print(y_test[0])

# summarize feature map shapes
for i in range(len(model.layers)):
	layer = model.layers[i]
	# check for convolutional layer
	if 'conv' not in layer.name:
		continue
	# summarize output shape
	print(i, layer.name, layer.output.shape)


print(x_test.shape)

# study the meaning of the filtered outputs by comparing them for
# multiple samples
nLastLayer = len(model.layers)-1
nLayer=nLastLayer
print("lastLayer:",nLastLayer)

baseFilenameForSave=None
x_test_flat=None
ViewTools_NN.printFeatureMapsForLayer(nLayer, model, x_test_flat, x_test, 0, baseFilenameForSave)
ViewTools_NN.printFeatureMapsForLayer(nLayer, model, x_test_flat, x_test, 1, baseFilenameForSave)
ViewTools_NN.printFeatureMapsForLayer(nLayer, model, x_test_flat, x_test, 2, baseFilenameForSave)
ViewTools_NN.printFeatureMapsForLayer(nLayer, model, x_test_flat, x_test, 3, baseFilenameForSave)
ViewTools_NN.printFeatureMapsForLayer(nLayer, model, x_test_flat, x_test, 4, baseFilenameForSave)
ViewTools_NN.printFeatureMapsForLayer(nLayer, model, x_test_flat, x_test, 6, baseFilenameForSave)
ViewTools_NN.printFeatureMapsForLayer(nLayer, model, x_test_flat, x_test, 5, baseFilenameForSave)
ViewTools_NN.printFeatureMapsForLayer(nLayer, model, x_test_flat, x_test, 7, baseFilenameForSave)
