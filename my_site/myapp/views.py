from django.shortcuts import render
from django.http import HttpResponse
from django.core.files.storage import FileSystemStorage
from django.conf import settings
import PyPDF2 as pdf
import textract as tt
import re
import nltk


def index(request):
    context = {'a':'Hello Word'}
    return render(request,'index.html',context)
# Create your views here.
articles =""
def fullarticle(request):
   context = {'article':''}
   if request.method == 'POST':
       temp={}
       temp['textArea']=request.POST.get('textArea')
       temp['file']=request.POST.get('file')
       context = {}
       clean_article=[]
       clean_textareaArticle=[]
       # doc = ''
       if request.POST.get('getSummary'):
            print(temp['textArea'])
            print(temp['file'])
            global articles
            
            if temp['textArea'] !="" and temp['file'] !="":
                temp['message'] ="You can choose just one choice .pdf or text"
            elif temp['textArea'] != "" and temp['file'] == "":
                textareaArticle =  temp['textArea']  
                clean_textareaArticle.append(cleanText(textareaArticle,removeStopwords=False))
                temp['CleantextArea'] = clean_textareaArticle 
            if temp['file'] is None and temp['textArea'] == "        ":  
                uploaded_file = request.FILES['file']
                print(uploaded_file.name)
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
                temp['article'] = clean_article  
   return render(request,'index.html',temp)

def pdf_To_text(text):
    # creating a pdf file object
    pdf_file = open(text, 'rb')
    # creating a pdf reader object
    read_pdf = pdf.PdfFileReader(pdf_file)
    # get number of pages in pdf file
    number_of_pages = read_pdf.getNumPages()
    #print the # of page
    print(number_of_pages)
    
    #read page by page with getPage method
    for i in range(0,number_of_pages):
        p = read_pdf.getPage(i)
        text += p.extractText()
        
    # text arrangement
    text = text.replace("\n", " ")
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

