# 递归获取路径下所有文件名1     
# -*- coding: utf-8 -*-  
import os 
#os.walk()
#os.listdir()

import shutil
        
def file_name(file_dir):   
    L=[]
    only_file_names=[]  
    for root, dirs, files in os.walk(file_dir):  
        #print(root) # 当前路径
        #print(dirs) #　当前目路
        #print(files) #　文件名
        for file in files:  
            if os.path.splitext(file)[1] == '.pdf':  # 想要保存的文件格式
                #print(os.path.splitext(file)[0])
                only_file_names.append(os.path.splitext(file)[0])
                #L.append(os.path.join(root, file)) # 返回带路径的文件名
        #print(only_file_names)
        #print(L)   
    return only_file_names #L  

 
#其中os.path.splitext()函数将路径拆分为文件名+扩展名

def find_object_name(input_txt_file):
    temp_id=[]
    #with open("test.txt", "r") as f:
    with open(input_txt_file, "r") as f:
        #youtube_id = f.read()  # 读取文件
        #youtube_id = f.readlines() #读取文件内容以数列格式返回
        for youtube_id in f.readlines():#youtube_id:
            youtube_id = youtube_id.strip('\n')
            temp_id.append(youtube_id)        
        #print(youtube_id)
        #print(id)
        #print(temp_id)
    return temp_id

def compare_file_name(input_list1,input_list2,output_txt_file):
    temp=[]
    for name in youtube_names:
        if name not in pdf_file_names:
            #print(name)
            temp.append(name)
    #print(temp)
    #output_txt_file='D:\\20210706E\\2020-python\\move_videos\\videos\\output_youtube_id.txt'
    with open(output_txt_file,"w") as f:
        #sep='\n'
        #f.write(sep.join(temp))  #
        for line in temp:
            f.writelines(line+'\n')


def mymovefile(srcfile,dstfile):
    if not os.path.isfile(srcfile):
        print ("%s not exist!"%(srcfile))
    else:
        fpath,fname=os.path.split(dstfile)    #分离文件名和路径
        if not os.path.exists(fpath):
            os.makedirs(fpath)                #创建路径
        shutil.move(srcfile,dstfile)          #移动文件
        print ("move %s -> %s"%( srcfile,dstfile))


def mycopyfile(srcfile,dstfile):
    if not os.path.isfile(srcfile):
        print ("%s not exist!"%(srcfile))
    else:
        fpath,fname=os.path.split(dstfile)    #分离文件名和路径
        if not os.path.exists(fpath):
            os.makedirs(fpath)                #创建路径
        shutil.copyfile(srcfile,dstfile)      #复制文件
        print ("copy %s -> %s"%( srcfile,dstfile))

if __name__=='__main__':

    #找到已经下载的youtube视频文件id
    work_dir = "D:\\20210706E\\2020-python\\move_videos"
    pdf_file_names=file_name(work_dir) # 已下载的文件
    print(pdf_file_names)

    #读入需要下载youtube视频文件id
    input_txt_file='D:\\20210706E\\2020-python\\move_videos\\videos\\youtube_id.txt'
    youtube_names=find_object_name(input_txt_file) #　列表需要下载的文件
    print(youtube_names)

    #找到未下载的youtube视频文件，并保存到txt文件中
    output_txt_file='D:\\20210706E\\2020-python\\move_videos\\videos\\output_youtube_id.txt'
    compare_file_name(pdf_file_names,youtube_names,output_txt_file)

    output_pdf='D:\\20210706E\\2020-python\\move_videos\\videos'

    for name in youtube_names:

        srcfile=os.path.join(work_dir,name+'.pdf')
        dstfile=os.path.join(output_pdf,name+'.pdf')

        mycopyfile(srcfile,dstfile)
        #mymovefile(srcfile,dstfile)
   