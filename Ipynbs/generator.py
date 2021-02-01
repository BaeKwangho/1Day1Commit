import os
import sys
import librosa
#import matplotlib.pyplot as plt
#import IPython.display as lpd

def load_audio(path):
    return librosa.load(path,sr=16000)[0]
#version 1
#files = []
#version 2

def gen():
    files = {}

    json = {}
    json_word_num=3
    json['<unk>'] = [0,1]
    json['<sos>'] = [1,1]
    json['<eos>'] = [2,1]
    
    folder_path = '/root/storage/DATA/LibriSpeech/dev-clean/'
    folder_1s = os.listdir(folder_path)
    for i,folder_1 in enumerate(folder_1s):
        folder1_path = os.path.join(folder_path,folder_1)
        folder_2s = os.listdir(folder1_path)
        for j,folder_2 in enumerate(folder_2s):
            folder2_path = os.path.join(folder1_path,folder_2)
            folder_3s = sorted(os.listdir(folder2_path))

            for k,file in enumerate(folder_3s):
                f_dic = {
                    'file_path':None,
                    'file_txt':None,
                }
                if file.split('.')[-1]=='txt':
                    with open(os.path.join(folder2_path,file))as f:
                        for line in f.readlines():
                            text = [text.lower() for text in line.split(' ')]
                            name = text[0]
                            text[0] = '<sos>'
                            text[-1] = text[-1][:-1]
                            text = text+['<eos>']

                            # "''" 제거
                            '''
                            def expand_contractions(text):
                                ret = []
                                for i in text:
                                    word = contractions.fix(i)
                                    ret.append(word)
                                return ret
                            text = expand_contractions(text)
                            '''

                            #version 2
                            if str(name) in files.keys():
                                files[str(name)]['file_txt'] = text

                            #json edit
                            for word in text:
                                if str(word) in json.keys():
                                    json[str(word)][1]+=1
                                else:
                                    json[str(word)] = [json_word_num,1]
                                    json_word_num += 1

                if file.split('.')[-1]=='wav':
                    filepath = os.path.join(folder2_path,file)
                    #version2
                    files[str(file.split('.')[0])] = {}
                    files[str(file.split('.')[0])]['file_path'] = filepath

                    #version1        
                    #f_dic['file_path']=filepath
                    #small_folder.append(f_dic)
    json_word = {}
    for i in json:
        json_word[json[i][0]]=i
        
    return files,json,json_word

def gen2():
    files = {}

    json = {}
    json_word_num=3
    json['<unk>'] = [0,1]
    json['<sos>'] = [1,1]
    json['<eos>'] = [2,1]
    
    folder_path = '/root/storage/DATA/LibriSpeech/dev-clean/'
    folder_1s = os.listdir(folder_path)
    for i,folder_1 in enumerate(folder_1s):
        folder1_path = os.path.join(folder_path,folder_1)
        folder_2s = os.listdir(folder1_path)
        for j,folder_2 in enumerate(folder_2s):
            folder2_path = os.path.join(folder1_path,folder_2)
            folder_3s = sorted(os.listdir(folder2_path))

            for k,file in enumerate(folder_3s):
                f_dic = {
                    'file_path':None,
                    'file_txt':None,
                }
                if file.split('.')[-1]=='txt':
                    with open(os.path.join(folder2_path,file))as f:
                        for line in f.readlines():
                            text = [word.lower() for word in line.split(' ')]
                            name = text[0]
                            text = text[1:]
                            text = ' '.join(text)
                            
                            # "''" 제거
                            '''
                            def expand_contractions(text):
                                ret = []
                                for i in text:
                                    word = contractions.fix(i)
                                    ret.append(word)
                                return ret
                            text = expand_contractions(text)
                            '''

                            #version 2
                            if str(name) in files.keys():
                                files[str(name)]['file_txt'] = text


                if file.split('.')[-1]=='wav':
                    filepath = os.path.join(folder2_path,file)
                    #version2
                    files[str(file.split('.')[0])] = {}
                    files[str(file.split('.')[0])]['file_path'] = filepath

                    #version1        
                    #f_dic['file_path']=filepath
                    #small_folder.append(f_dic)
    json_word = {}
    for i in json:
        json_word[json[i][0]]=i
        
    return files