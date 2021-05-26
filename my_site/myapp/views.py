from django.shortcuts import render
from django.http import HttpResponse
from django.core.files.storage import FileSystemStorage
from django.conf import settings
import PyPDF2 as pdf
import re
import numpy as np
import re
from keras.preprocessing.text import Tokenizer 
from keras.preprocessing.sequence import pad_sequences       
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense, Concatenate, TimeDistributed
from tensorflow.keras.models import Model
from keras import backend as K 
from .attention import AttentionLayer



def index(request):
    context = {'a':'Hello World'}
    return render(request,'index.html',context)
# Create your views here.
articles = ""
temp={}

count =0
def fullarticle(request):
   context = {'article':''}


   if request.method == 'POST':
       


       temp['textArea']=request.POST.get('textArea')
       temp['file']=request.POST.get('file')
       context = {}
       clean_article=[]
       clean_textareaArticle=[]
       result = []
       full_article=[]

       # if click the summary button evet
       if request.POST.get('getSummary'):
            #print(temp['textArea'])
            #print(temp['file'])
            global articles
            global count
            
            #if textarea and .pdf file is full condition
            if temp['textArea'] !="" and temp['file'] is None:
                temp['message'] ="You can choose just one choice .pdf or text"
            #if textarea is full
            if temp['textArea'] != "" and temp['file'] == "":
                textareaArticle =  temp['textArea']  
                clean_textareaArticle.append(cleanText(textareaArticle,removeStopwords=False))
                global count
                #split the article for model
                for i in range(0, len(clean_textareaArticle[0]), 3000):
                    slice_item = slice(i,i+3000)
                    result.append(clean_textareaArticle[0][slice_item])
                    count=count+1
                temp['textAreaNew'] = clean_textareaArticle
                #count = count - 1

            # if textarea is null .pdf is upload 
            if temp['file'] is None and temp['textArea'] == "":  
                uploaded_file = request.FILES['file']
                print(uploaded_file.name+" inside if")
                fs = FileSystemStorage()
                print(fs)
                global name
                name = fs.save(uploaded_file.name, uploaded_file)
                global doc
                doc = settings.MEDIA_ROOT + name
                print(doc)
                context['context'] = fs.url(name)
                
                articles = pdf_To_text(doc)
                clean_article.append(cleanText(articles, removeStopwords=False))
                
                #split the article for model
                for i in range(0, len(clean_article[0]), 3000):
                    slice_item = slice(i,i+3000)
                    result.append(clean_article[0][slice_item])
                    count=count+1
                temp['newpdf'] = clean_article
                #count = count - 1

            #preparing a tokenizer for summary on training data 
            text_tokenizer = Tokenizer()
            text_tokenizer.fit_on_texts(list(result))
            #convert summary sequences into integer sequences
            result    =   text_tokenizer.texts_to_sequences(result) 
            #padding zero upto maximum length
            result    =   pad_sequences(result, maxlen=600, padding='post')
            text_voc_size  =   len(text_tokenizer.word_index) +1    




            latent_dim = 220
            #embedding_dim=200
            max_text_length=600
            # Encoder
            encoder_inputs = Input(shape=(max_text_length,))

            #embedding layer
            enc_emb =  Embedding(text_voc_size, latent_dim,trainable=True)(encoder_inputs)

            #encoder lstm 1
            encoder_lstm1 = LSTM(latent_dim,return_sequences=True,return_state=True,dropout=0.4,recurrent_dropout=0.4)
            encoder_output1, state_h1, state_c1 = encoder_lstm1(enc_emb)

            #encoder lstm 2
            encoder_lstm2 = LSTM(latent_dim,return_sequences=True,return_state=True,dropout=0.4,recurrent_dropout=0.4)
            encoder_output2, state_h2, state_c2 = encoder_lstm2(encoder_output1)

            #encoder lstm 3
            encoder_lstm3=LSTM(latent_dim, return_state=True, return_sequences=True,dropout=0.4,recurrent_dropout=0.4)
            encoder_outputs, state_h, state_c= encoder_lstm3(encoder_output2)

            # Set up the decoder, using `encoder_states` as initial state.
            decoder_inputs = Input(shape=(None,))

            #embedding layer
            dec_emb_layer = Embedding(text_voc_size, latent_dim,trainable=True)
            dec_emb = dec_emb_layer(decoder_inputs)

            decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True,dropout=0.4,recurrent_dropout=0.4)
            decoder_outputs,decoder_fwd_state, decoder_back_state = decoder_lstm(dec_emb,initial_state=[state_h, state_c])

            # Attention layer
            attn_layer = AttentionLayer(name='attention_layer')
            attn_out, attn_states = attn_layer([encoder_outputs, decoder_outputs])

            # Concat attention input and decoder LSTM output
            decoder_concat_input = Concatenate(axis=-1, name='concat_layer')([decoder_outputs, attn_out])

            #dense layer
            decoder_dense =  TimeDistributed(Dense(text_voc_size, activation='softmax'))
            decoder_outputs = decoder_dense(decoder_concat_input)

            # Define the model 
            model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
            
            del model

            from tensorflow import keras
            model = keras.models.load_model('static/Saved_Model/my_model')

            

            reverse_target_word_index=text_tokenizer.index_word 
            #reverse_source_word_index=text_tokenizer.index_word 
            target_word_index=text_tokenizer.word_index

            #target_word_index[' '] = 0
            #reverse_target_word_index[0] = ' '


            
            # encoder inference
            encoder_model = Model(inputs=encoder_inputs,outputs=[encoder_outputs, state_h, state_c])
            # decoder inference
            # Below tensors will hold the states of the previous time step
            decoder_state_input_h = Input(shape=(latent_dim,))
            decoder_state_input_c = Input(shape=(latent_dim,))
            decoder_hidden_state_input = Input(shape=(max_text_length,latent_dim))

            # Get the embeddings of the decoder sequence
            dec_emb2= dec_emb_layer(decoder_inputs)

            # To predict the next word in the sequence, set the initial states to the states from the previous time step
            decoder_outputs2, state_h2, state_c2 = decoder_lstm(dec_emb2, initial_state=[decoder_state_input_h, decoder_state_input_c])

            #attention inference
            attn_out_inf, attn_states_inf = attn_layer([decoder_hidden_state_input, decoder_outputs2])
            decoder_inf_concat = Concatenate(axis=-1, name='concat')([decoder_outputs2, attn_out_inf])

            # A dense softmax layer to generate prob dist. over the target vocabulary
            decoder_outputs2 = decoder_dense(decoder_inf_concat)

            # Final decoder model
            decoder_model = Model(
            [decoder_inputs] + [decoder_hidden_state_input,decoder_state_input_h, decoder_state_input_c],
            [decoder_outputs2] + [state_h2, state_c2])




            
            max_summary_length=68
            def decode_sequence(input_seq):
                # Encode the input as state vectors.
                e_out, e_h, e_c = encoder_model.predict(input_seq)

                # Generate empty target sequence of length 1.
                target_seq = np.zeros((1,1))

                # Chose the 'start' word as the first word of the target sequence
                target_seq[0, 0] = target_word_index[reverse_target_word_index[input_seq[0][0]]]
                stop_condition = False
                decoded_sentence = ''
                SampledCompere = ''
                sampled_token = ''
                while not stop_condition:
                    output_tokens, h, c = decoder_model.predict([target_seq] + [e_out, e_h, e_c])
                    
                    # Sample a token
                    sampled_token_index = np.argmax(output_tokens[0, -1, :])

                    SampledCompere = sampled_token

                    if sampled_token_index == 0 :
                        break

                    sampled_token = reverse_target_word_index[sampled_token_index]

                    if SampledCompere == sampled_token:
                        sampled_token = ''

                    if(sampled_token!='end'):
                        decoded_sentence += ' '+sampled_token
                        
                        # Exit condition: either hit max length or find stop word.
                        if (sampled_token == 'end' or len(decoded_sentence.split()) >= (max_summary_length-1)):
                            stop_condition = True

                    # Update the target sequence (of length 1).
                    target_seq = np.zeros((1,1))
                    target_seq[0, 0] = sampled_token_index

                    # Update internal states
                    e_h, e_c = h, c

                decoded_sentence += '.'
                
                return decoded_sentence

        
            
            for i in range(0,count):
                full_article.append(decode_sequence(result[i].reshape(1,max_text_length)))
            temp['article'] = full_article

   return render(request,'index.html',temp)

def pdf_To_text(text):
    # creating a pdf file object
    pdf_file = open(text, 'rb')
    # creating a pdf reader object
    read_pdf = pdf.PdfFileReader(pdf_file)
    # get number of pages in pdf file
    number_of_pages = read_pdf.getNumPages()
    
    #read page by page with getPage method
    for i in range(0,number_of_pages):
        p = read_pdf.getPage(i)
        text += p.extractText()
        
    # text arrangement
    text = text.replace("\n", "")
    return text


# We must prepering the our data now
# A list of contractions 
contractions = { 
"ain't": "am not","aren't": "are not","can't": "cannot","can't've": "cannot have","cause": "because","could've": "could have","couldn't": "could not","couldn't've": "could not have","didn't": "did not","doesn't": "does not",
"don't": "do not","hadn't": "had not","hadn't've": "had not have","hasn't": "has not","haven't": "have not","he'd": "he would","he'd've": "he would have","he'll": "he will","he's": "he is","how'd": "how did","how'll": "how will","how's": "how is",
"i'd": "i would","i'll": "i will","i'm": "i am","i've": "i have","isn't": "is not","it'd": "it would","it'll": "it will","it's": "it is","let's": "let us","ma'am": "madam","mayn't": "may not","might've": "might have","mightn't": "might not","must've": "must have","mustn't": "must not",
"needn't": "need not","oughtn't": "ought not","shan't": "shall not","sha'n't": "shall not","she'd": "she would","she'll": "she will","she's": "she is",
"should've": "should have","shouldn't": "should not","that'd": "that would","that's": "that is","there'd": "there had","there's": "there is","they'd": "they would","they'll": "they will","they're": "they are","they've": "they have","wasn't": "was not","we'd": "we would","we'll": "we will","we're": "we are",
"we've": "we have","weren't": "were not","what'll": "what will","what're": "what are","what's": "what is","what've": "what have","where'd": "where did","where's": "where is","who'll": "who will","who's": "who is","won't": "will not","wouldn't": "would not","you'd": "you would","you'll": "you will","you're": "you are"
}

def cleanText(text, removeStopwords = True):
    '''Remove unwanted characters, stopwords, and format the text to create fewer nulls word embeddings'''
    
    # Convert words to lower case
    text = text.lower()
    
    # Replace contractions with their longer forms 
    if True:
        text = text.split()
        newText = []
        for word in text:
            if word in contractions:
                newText.append(contractions[word])
            else:
                newText.append(word)
        text = " ".join(newText)
    
    # Format words and remove unwanted characters
    text = re.sub(r'https?:\/\/.*[\r\n]*', '', text, flags=re.MULTILINE)
    text = re.sub(r'\<a href', ' ', text)
    text = re.sub(r'&amp;', '', text) 
    text = re.sub(r'[_"\-;%()|+&=*%.,!?:#$@\[\]/]', ' ', text)
    text = re.sub(r'<br />', ' ', text)
    text = re.sub(r'\'', ' ', text)
    
    return text

