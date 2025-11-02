"""## >Imports"""

import pandas as pd
import numpy as np
import cv2
import torch
from ultralytics import YOLO
import sys
import os
from pdf2image import convert_from_path
import regex
import img2pdf
import pytesseract
from openai import OpenAI, APIConnectionError
import threading
import time
import string, random
import gdown
from app.agents.ExtractAgent import ExtractAgent



#Creating this object will create necessary directories in the path given.
    #The /documents/ folder will be where pdf and loose image files will be processed.
        #If you do not want these files to be deleted pass clear=False to extract.
    #The weights for the yolo model will be downloaded into the folder given.
    #Use the extract method to get a list of indexes ready for upserting into Pinecone
        #The two indexes right now are 'paragraph' and 'captioned image' in that order.
        #They can be upserted into their own namespaces.
        #The metadata within these indexes will contain 
            #'text' : the original text 
            #'image': absolute path to the image file that produced that text.
class YoloExtractor:

    #path is the root directory that all the processing will happen in.
        #This directory should have a copy of documents.pt
    def __init__(self, path, extract:ExtractAgent):
        self.extract_agent = extract
        """## >Constants"""
        self.URL = 'https://drive.google.com/file/d/18EEqktXDPWWOROY1Fc2WbFu825QTScBK/view?usp=drive_link'
        self.PATH = path +'/'
        self.FILE_PATH = path + '/documents/'
        self.PROJECT_NAME = '/document run/'
        self.IMAGES_PATH = path + '/images/'
        self.TEMP_PATH = path + '/temp/'
        self.WEIGHTS_NAME = path + '/documents.pt'
        self.PROJECT_PATH = path + self.PROJECT_NAME

        """Create Directories"""
        if(not os.path.exists(self.PATH)):
            os.mkdir(self.PATH)
        if(not os.path.exists(self.FILE_PATH)):
            os.mkdir(self.FILE_PATH)
        if(not os.path.exists(self.IMAGES_PATH)):
            os.mkdir(self.IMAGES_PATH)
        if(not os.path.exists(self.IMAGES_PATH+'paragraph')):
            os.mkdir(self.IMAGES_PATH+'paragraph')
        if(not os.path.exists(self.IMAGES_PATH+'captioned image')):
            os.mkdir(self.IMAGES_PATH+'captioned image')
        if(not os.path.exists(self.TEMP_PATH)):
            os.mkdir(self.TEMP_PATH)
        if(not os.path.exists(self.TEMP_PATH+'paragraph')):
            os.mkdir(self.TEMP_PATH+'paragraph')
        if(not os.path.exists(self.TEMP_PATH+'captioned image')):
            os.mkdir(self.TEMP_PATH+'captioned image')
        if(not os.path.exists(self.PROJECT_PATH)):
            os.mkdir(self.PROJECT_PATH)
        


        # self.context = dict()

        self.IMAGE_SIZE = 640
        self.FORMATS = ['.bmp','.dng','.jpeg',
                '.jpg','.mpo','.png',
                '.tif','.tiff','.webp',
                '.pfm','.HEIC']

        self.CLASSES =   {0 : 'captioned image',
                    1 : 'figure',
                    2 : 'formula',
                    3 : 'paragraph',
                    4 : 'reference',
                    5 : 'title'}

        if(torch.cuda.is_available()):
            print('GPU Enabled')
            self.GPU = 0
            self.WORKERS = 2
        else:
            print('GPU Disabled')
            self.GPU = None
            self.WORKERS = 8
        
        """# Ultralytics YOLO model"""
        if(os.path.isfile(self.WEIGHTS_NAME)):
            self.model = YOLO(self.WEIGHTS_NAME)
            print('Loaded Weights at: ' + self.WEIGHTS_NAME)
        else:
            #Download the weights.
            gdown.download(self.URL, self.WEIGHTS_NAME, quiet=False, fuzzy=True)
            self.model = YOLO(self.WEIGHTS_NAME)
            print('Loaded Weights at: ' + self.WEIGHTS_NAME)
        
        self.indexes = []
        self.client = OpenAI()

        self.paragraphDataFrame = pd.DataFrame(columns=['id', 'values', 'metadata'])
        self.captionedImageDataFrame = pd.DataFrame(columns=['id', 'values', 'metadata'])


    """## >Methods"""

    """Yolo Extraction Methods"""

    def __read_bounding_boxes(self, results, name):
        for i, result in enumerate(results):
            image = result.orig_img
            for j, cls in enumerate(result.boxes.cls.tolist()):
                if(self.CLASSES[cls] == 'paragraph' or self.CLASSES[cls] == 'captioned image'):
                    x1, y1, x2, y2 = np.array(result.boxes.xyxy[j].cpu(), dtype=np.int64)
                    path = str(self.TEMP_PATH+'/'+self.CLASSES[cls]+'/'+name+'-pg'+str(i)+',box'+str(j)+'.png')
                    cv2.imwrite(path, image[y1:y2, x1:x2])

    def __read_pdf(self, pdf):
        images = convert_from_path(pdf)
        return images

    def __run_yolo(self, folderpath, images, name):
        results = self.model.predict(images, save=True, name=name, project = self.PROJECT_PATH,verbose=False,
                                device=self.GPU, conf=.15, iou=.05, exist_ok=True)
        self.__read_bounding_boxes(results, name)

    def __clear_folder(self, folderpath):
        for i in os.listdir(folderpath):
            if(not os.path.isdir(folderpath+'/'+i)):
                os.remove(folderpath+'/'+i)

    def __detect_documents(self, folderpath, clear):
        pdfs = []
        names = []
        loose_images = []
        for i in os.listdir(folderpath):
            if(i[-4:] in self.FORMATS or i[-5:] in self.FORMATS):
                loose_images.append(cv2.imread(folderpath+i))
                sys.stdout.write(f"\rLoose Images: {len(loose_images)}")
                sys.stdout.flush()
        print('')
        for i in os.listdir(folderpath):
            if(i[-4:] == '.pdf'):
                images = self.__read_pdf(folderpath+i)
                pdfs.append(images)
                names.append(i)
                sys.stdout.write(f"\rPDFs: {len(pdfs)}")
                sys.stdout.flush()
        assert not (len(loose_images) == 0 and len(pdfs) == 0), 'No Files Found!'
        print('\nFiles Loaded.')
        if(len(loose_images) > 0):
            self.__run_yolo(folderpath, loose_images, 'loose.pdf')
        for i,images in enumerate(pdfs):
            self.__run_yolo(folderpath ,images, names[i])
        if(clear):
            self.__clear_folder(folderpath)


    """Compile Annotated Images Into PDFs"""

    def __compile_images(self):
        pdfs = []
        for pdf in os.listdir(self.PROJECT_PATH):
            temp = [pdf]
            paths = []
            for i in os.listdir(self.PROJECT_PATH+pdf):
                if(regex.search("e[1-9].jpg", i)):
                    i_sub = regex.sub("e","e0",i)
                    os.rename(self.PROJECT_PATH+pdf+'/'+i, self.PROJECT_PATH+pdf+'/'+i_sub)
                    i = i_sub
                if(i[-4:] == '.jpg'):
                    paths.append(self.PROJECT_PATH+pdf+'/'+i)
            temp.append(sorted(paths))
            pdfs.append(temp)
        return pdfs

    def __compile_pdfs(self):
        pdfs = self.__compile_images()
        for pdf in pdfs:
            with open(self.PATH+pdf[0], "wb") as f:
                if(len(pdf[1]) > 0):
                    f.write(img2pdf.convert(pdf[1]))
                    print('Recompiled: '+self.PATH+pdf[0]+ '!')
            self.__clear_folder(self.PROJECT_PATH+pdf[0])

    """Image Parsing"""

    def __get_embeddings(self, text, model="text-embedding-3-small"):
        return self.client.embeddings.create(input=text, model=model).data[0].embedding

    def __generate_ids(self, number, size):
        import string, random
        ids=[]
        for i in range(number):
            res = ''.join(random.choices(string.ascii_letters, k=size))
            ids.append(res)
            if len(set(ids)) != i+1:
                i-=1
                ids.pop(-1)
        return ids

    def __clean_text(self, text):
        text = regex.sub(r'\[[^]]+\]', '', text)
        text = regex.sub(r'[\n\x0c]',' ',text)
        text = regex.sub(r'/4Z', '', text)
        text = regex.sub(r'[0-9(][0-9][)\]]', '', text)
        return regex.sub(r'[^a-z,A-Z0-9.:;()  \'\"]','',text)
    
    def __load_chunk(self, df, id, chunk, image, i, folder):
        if(folder == 'captioned image' or self.extract_agent.analyze(chunk)):
            df.loc[i] = [id,
                self.__get_embeddings(chunk,
                                model='text-embedding-3-small'),
                                    {'text': chunk,
                                    'image': image
                                    }
                ]
        else:
            df.loc[i] = [id,pd.NA,pd.NA]

    def __read_worker(self, paths, folder, ids, df, i_list):
        for i,image in enumerate(paths):
            text = self.__clean_text(
                                    pytesseract.image_to_string(
                                        cv2.imread(self.TEMP_PATH + '/' + folder + '/'+image))
                                    )
            image_path = self.IMAGES_PATH + folder + '/'+ image
            try:
                self.__load_chunk(df, ids[i], text, image_path, i_list[i], folder)
            except APIConnectionError as e:
                print('Api Connection Error')
                time.sleep(5)
                self.__read_worker(paths[i:], folder, ids[i:], df, i_list[i:])
                return
            sys.stdout.write(
                f"\r{folder} Parsed: {len(df)} / {len(os.listdir(self.TEMP_PATH + '/' + folder))}"
                )
            sys.stdout.flush()
        time.sleep(1)

    def __read_images_parallel(self):
        dfs = {'paragraph':self.paragraphDataFrame, 'captioned image':self.captionedImageDataFrame}
        for folder in os.listdir(self.TEMP_PATH):
            directories = os.listdir(self.TEMP_PATH + '/' + folder)
            nd = len(directories)
            ids = self.__generate_ids(nd,10)
            if(nd != 0):
                n = 8
                if(nd//n == 0):
                    n = nd-1
                if(n == 0):
                    n = 1
                print((0,nd,nd//n,n))
                x = list(range(0,nd,nd//n))
                threads = []
                for i in range(0,n):
                    if(i == n-1):
                        chunk = directories[x[i]:]
                        id_chunk = ids[x[i]:]
                        i_list = range(x[i],nd+1)
                    else:
                        chunk = directories[x[i]:x[i+1]]
                        id_chunk = ids[x[i]:x[i+1]]
                        i_list = range(x[i],x[i+1]+1)
                    threads.append(threading.Thread(
                                                    target=self.__read_worker,
                                                    args=(chunk,
                                                        folder,
                                                        id_chunk,
                                                        dfs[folder],
                                                        i_list
                                                        )
                                                    )
                                )
                    threads[i].start()
                for thread in threads:
                    thread.join()
                print('')

    """Run Extraction Method"""
    def extract(self, printPDF=True, clear=True):
        self.__detect_documents(self.FILE_PATH, clear=clear)
        
        if(printPDF):
            self.__compile_pdfs()
            print('PDFs Recompiled!')

        self.__read_images_parallel()

        self.paragraphDataFrame.dropna(inplace=True)
        self.captionedImageDataFrame.dropna(inplace=True)

        self.indexes.append(self.paragraphDataFrame)
        self.indexes.append(self.captionedImageDataFrame)

        self.paragraphDataFrame.to_csv(self.PATH+'paragraph.csv',index=False)
        self.captionedImageDataFrame.to_csv(self.PATH+'captioned image.csv',index=False)

        return self.indexes