from django.shortcuts import render
from .models import ImageUpload
from .forms import ImageForm
from .apps import TokenizerConfig , CaptionModelConfig , FeatureExtModelConfig
from keras.applications.vgg16 import preprocess_input
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.image import load_img , img_to_array

# Create your views here.

tokenizer=TokenizerConfig.tokenzizer
feature_ext=FeatureExtModelConfig.feature_extractor
model=CaptionModelConfig.model


def idx_to_word(integer , tokenizer):
    for word , index in tokenizer.word_index.items():
        if index == integer:
            return word

    return None


def predict_caption(model , image , tokenizer , max_length):
    # adding start tag
    in_text = 'startseq'
    #iterate over max length on sequence
    for i in range(max_length):
        #encode input sequence
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        # padding
        sequence = pad_sequences([sequence] , max_length)
        # predict next word
        y = model.predict([image , sequence] , verbose=0)
        # greedy decoding
        y = np.argmax(y)
        # index to word
        word = idx_to_word(y , tokenizer)
        #stop if word not found
        if word is None:
            break
        # append for next input
        in_text = in_text + " " + word
        # stop if word is <end>
        if word == "endseq":
            break
    
    return in_text


def CaptionGen(img):

    image = load_img(img , target_size=(224,224))
    image = img_to_array(image)
    image = image.reshape(1 , image.shape[0] , image.shape[1] , image.shape[2])
    image = preprocess_input(image)

    features = feature_ext(image)
    caption = predict_caption(model , features , tokenizer ,35)

    return caption

def clean_caption(caption):
    cap_list = caption.split()
    clean_captions = ''
    for i in range(1 , len(cap_list) - 1):
        clean_captions = clean_captions + ' ' + cap_list[i]

    return clean_captions    

def captionView(request):

    if request.method=='POST':
        form = ImageForm(request.POST , request.FILES)

        if form.is_valid():
            form.save()
            image_obj = form.instance
            image = image_obj.image
            cap = CaptionGen('media/' + str(image))
            clean_cap = clean_caption(cap)
            
            context = {
                'form' : form,
                'image': image,
                'caption':clean_cap,
            }
            return render(request ,'main.html' , context=context)
    
    form = ImageForm()

    return render(request ,'main.html' , {'form':form})