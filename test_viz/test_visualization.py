import numpy as np
import tensorflow as tf
from pathlib import Path
import pandas as pd
from hieralb.core import Album
from psiz.visualize import visualize_embedding_static

ALBUM_PATH = Path('/Users/bdroads/Dropbox') / Path('exp-datasets', 'birds-12')

APP_PATH = Path('birds_12')

def main():
    '''
    '''

    album = Album(ALBUM_PATH)

    album_info = pd.read_csv(APP_PATH / Path('album_info.txt'))
    embedding = pd.read_csv(APP_PATH / Path('embedding.txt'), sep=' ', names=('id','dim0','dim1','dim2'))
    
    Z = embedding.as_matrix()
    Z_2D = Z[:,0:2]
    Z_3D = Z[:,0:3]
    print(Z_3D.shape)

    # Clean up labels
    clean_classes = {k: pretty_label(v) for k, v in album.classes.items()}
    
    visualize_embedding_static(Z_2D, filename='psiz-app/test_0.pdf')
    visualize_embedding_static(Z_2D, class_vec=album_info.class_id_0, filename='psiz-app/test_1.pdf')
    visualize_embedding_static(Z_2D, class_vec=album_info.class_id_0, classes = clean_classes, filename='psiz-app/test_2.pdf')

    visualize_embedding_static(Z_3D, filename='psiz-app/test_3.pdf')
    visualize_embedding_static(Z_3D, class_vec=album_info.class_id_0, filename='psiz-app/test_4.pdf')
    visualize_embedding_static(Z_3D, class_vec=album_info.class_id_0, classes = clean_classes, filename='psiz-app/test_5.pdf')
    # visualize_embedding_static(Z_3D, class_vec=album_info.class_id_0, classes = clean_classes)

def pretty_label(dirty_str):
    '''Returns an altered string. The returned string has a upper case first 
    character, underscores replaced with spaces, and characeters following
    spaces are upper case.
    '''

    dirty_str = index_replace(dirty_str, 0, str.upper(dirty_str[0]))
    l = len(dirty_str)

    is_dirty = True
    while is_dirty:
        idx = dirty_str.find('_')
        if idx is -1:
            is_dirty = False
        else:
            dirty_str = index_replace(dirty_str, idx, ' ')
            if idx+1 < l:
                dirty_str = index_replace(dirty_str, idx+1, str.upper(dirty_str[idx+1]))
    return dirty_str

def index_replace(word, index, char):
    '''Since strings are immutable, a handy function for replacing characters
    at specific indices.

    see: https://stackoverflow.com/questions/12723751/replacing-instances-of-a-character-in-a-string
    '''

    word = word[:index] + char + word[index + 1:]
    return word

if __name__ == "__main__":
    main()
