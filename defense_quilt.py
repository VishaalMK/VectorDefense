import PIL
import PIL.Image
from io import BytesIO
import numpy as np
import tensorflow as tf


def make_defend_quilt(sess):
    # setup for quilting
    quilt_db = np.load('data/mnist_quilt_db.npy')
    quilt_db_reshaped = quilt_db.reshape(60000, -1)
    TILE_SIZE = 4
    TILE_OVERLAP = 2
    tile_skip = TILE_SIZE - TILE_OVERLAP
    K = 10
    db_tensor = tf.placeholder(tf.float32, quilt_db_reshaped.shape)
    query_imgs = tf.placeholder(tf.float32, (TILE_SIZE * TILE_SIZE * 1, None))
    norms = tf.reduce_sum(tf.square(db_tensor), axis=1)[:, tf.newaxis] \
                    - 2*tf.matmul(db_tensor, query_imgs)
    _, topk_indices = tf.nn.top_k(-tf.transpose(norms), k=K, sorted=False)
    def min_error_table(arr, direction):
        assert direction in ('horizontal', 'vertical')
        y, x = arr.shape
        cum = np.zeros_like(arr)
        if direction == 'horizontal':
            cum[:, -1] = arr[:, -1]
            for ix in range(x-2, -1, -1):
                for iy in range(y):
                    m = arr[iy, ix+1]
                    if iy > 0:
                        m = min(m, arr[iy-1, ix+1])
                    if iy < y - 1:
                        m = min(m, arr[iy+1, ix+1])
                    cum[iy, ix] = arr[iy, ix] + m
        elif direction == 'vertical':
            cum[-1, :] = arr[-1, :]
            for iy in range(y-2, -1, -1):
                for ix in range(x):
                    m = arr[iy+1, ix]
                    if ix > 0:
                        m = min(m, arr[iy+1, ix-1])
                    if ix < x - 1:
                        m = min(m, arr[iy+1, ix+1])
                    cum[iy, ix] = arr[iy, ix] + m
        return cum
    def index_exists(arr, index):
        if arr.ndim != len(index):
            return False
        return all(i > 0 for i in index) and all(index[i] < arr.shape[i] for i in range(arr.ndim))
    def assign_block(ix, iy, tile, synth):
        posx = tile_skip * ix
        posy = tile_skip * iy

        if ix == 0 and iy == 0:
            synth[posy:posy+TILE_SIZE, posx:posx+TILE_SIZE, :] = tile
        elif iy == 0:
            # first row, only have horizontal overlap of the block
            tile_left = tile[:, :TILE_OVERLAP, :]
            synth_right = synth[:TILE_SIZE, posx:posx+TILE_OVERLAP, :]
            errors = np.sum(np.square(tile_left - synth_right), axis=2)
            table = min_error_table(errors, direction='vertical')
            # copy row by row into synth
            xoff = np.argmin(table[0, :])
            synth[posy, posx+xoff:posx+TILE_SIZE] = tile[0, xoff:]
            for yoff in range(1, TILE_SIZE):
                # explore nearby xoffs
                candidates = [(yoff, xoff), (yoff, xoff-1), (yoff, xoff+1)]
                index = min((i for i in candidates if index_exists(table, i)), key=lambda i: table[i])
                xoff = index[1]
                synth[posy+yoff, posx+xoff:posx+TILE_SIZE] = tile[yoff, xoff:]
        elif ix == 0:
            # first column, only have vertical overlap of the block
            tile_up = tile[:TILE_OVERLAP, :, :]
            synth_bottom = synth[posy:posy+TILE_OVERLAP, :TILE_SIZE, :]
            errors = np.sum(np.square(tile_up - synth_bottom), axis=2)
            table = min_error_table(errors, direction='horizontal')
            # copy column by column into synth
            yoff = np.argmin(table[:, 0])
            synth[posy+yoff:posy+TILE_SIZE, posx] = tile[yoff:, 0]
            for xoff in range(1, TILE_SIZE):
                # explore nearby yoffs
                candidates = [(yoff, xoff), (yoff-1, xoff), (yoff+1, xoff)]
                index = min((i for i in candidates if index_exists(table, i)), key=lambda i: table[i])
                yoff = index[0]
                synth[posy+yoff:posy+TILE_SIZE, posx+xoff] = tile[yoff:, xoff]
        else:
            # glue cuts along diagonal
            tile_up = tile[:TILE_OVERLAP, :, :]
            synth_bottom = synth[posy:posy+TILE_OVERLAP, :TILE_SIZE, :]
            errors_up = np.sum(np.square(tile_up - synth_bottom), axis=2)
            table_up = min_error_table(errors_up, direction='horizontal')
            tile_left = tile[:, :TILE_OVERLAP, :]
            synth_right = synth[:TILE_SIZE, posx:posx+TILE_OVERLAP, :]
            errors_left = np.sum(np.square(tile_left - synth_right), axis=2)
            table_left = min_error_table(errors_left, direction='vertical')
            glue_index = -1
            glue_value = np.inf
            for i in range(TILE_OVERLAP):
                e = table_up[i, i] + table_left[i, i]
                if e < glue_value:
                    glue_value = e
                    glue_index = i
            # copy left part first, up to the overlap column
            xoff = glue_index
            synth[posy+glue_index, posx+xoff:posx+TILE_OVERLAP] = tile[glue_index, xoff:TILE_OVERLAP]
            for yoff in range(glue_index+1, TILE_SIZE):
                # explore nearby xoffs
                candidates = [(yoff, xoff), (yoff, xoff-1), (yoff, xoff+1)]
                index = min((i for i in candidates if index_exists(table_left, i)), key=lambda i: table_left[i])
                xoff = index[1]
                synth[posy+yoff, posx+xoff:posx+TILE_OVERLAP] = tile[yoff, xoff:TILE_OVERLAP]
            # copy right part, down to overlap row
            yoff = glue_index
            synth[posy+yoff:posy+TILE_OVERLAP, posx+glue_index] = tile[yoff:TILE_OVERLAP, glue_index]
            for xoff in range(glue_index+1, TILE_SIZE):
                # explore nearby yoffs
                candidates = [(yoff, xoff), (yoff-1, xoff), (yoff+1, xoff)]
                index = min((i for i in candidates if index_exists(table_up, i)), key=lambda i: table_up[i])
                yoff = index[0]
                synth[posy+yoff:posy+TILE_OVERLAP, posx+xoff] = tile[yoff:TILE_OVERLAP, xoff]
            # copy rest of image
            synth[posy+TILE_OVERLAP:posy+TILE_SIZE, posx+TILE_OVERLAP:posx+TILE_SIZE] = tile[TILE_OVERLAP:, TILE_OVERLAP:]
    KNN_MAX_BATCH = 1000
    def quilt(arr, graphcut=True):
        h, w, c = arr.shape
        assert (h - TILE_SIZE) % tile_skip == 0
        assert (w - TILE_SIZE) % tile_skip == 0
        horiz_blocks = (w - TILE_SIZE) // tile_skip + 1
        vert_blocks = (h - TILE_SIZE) // tile_skip + 1
        num_patches = horiz_blocks * vert_blocks
        patches = np.zeros((TILE_SIZE * TILE_SIZE * 1, num_patches))
        idx = 0
        for iy in range(vert_blocks):
            for ix in range(horiz_blocks):
                posx = tile_skip*ix
                posy = tile_skip*iy
                patches[:, idx] = arr[posy:posy+TILE_SIZE, posx:posx+TILE_SIZE, :].ravel()
                idx += 1
        
        ind = []
        for chunk in range(num_patches // KNN_MAX_BATCH + (1 if num_patches % KNN_MAX_BATCH != 0 else 0)):
            start = KNN_MAX_BATCH * chunk
            end = start + KNN_MAX_BATCH
            # for some reason, the code below is 10x slower when run in a Jupyter notebook
            # not sure why...
            indices_ = sess.run(topk_indices, {db_tensor: quilt_db_reshaped, query_imgs: patches[:, start:end]})
            for i in indices_:
                ind.append(np.random.choice(i))
        
        synth = np.zeros((28, 28, 1))
        
        idx = 0
        for iy in range(vert_blocks):
            for ix in range(horiz_blocks):
                posx = tile_skip*ix
                posy = tile_skip*iy
                tile = quilt_db[ind[idx]]
                if not graphcut:
                    synth[posy:posy+TILE_SIZE, posx:posx+TILE_SIZE, :] = tile
                else:
                    assign_block(ix, iy, tile, synth)
                idx += 1
        return synth
    
    return quilt
