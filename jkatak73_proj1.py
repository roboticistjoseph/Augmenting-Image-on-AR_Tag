## Importing Necessary Library
import numpy as np # for Array Operations
import cv2 # for Computer Vision Applications
import scipy.fftpack # for Fourier Transformation

# Using Fast Fourier Transform to perform blurring operation using Gaussian Blur
def blur_img(image_gray):

    fft_image = scipy.fft.fft2(image_gray, axes = (0,1))
    fft_image_shifted = scipy.fft.fftshift(fft_image)
    
    kernel_x = 40
    kernel_y = 40
    cols, rows = image_gray.shape
    center_x, center_y = rows / 2, cols / 2
    rows = np.linspace(0, rows, rows)
    cols = np.linspace(0, cols, cols)
    X, Y = np.meshgrid(rows, cols)
    Gmask = np.exp(-(np.square((X - center_x)/kernel_x) + np.square((Y - center_y)/kernel_y)))
    
    
    fft_image_blur = fft_image_shifted * Gmask


    img_shifted_back = scipy.fft.ifftshift(fft_image_blur)
    img_back_blur = scipy.fft.ifft2(img_shifted_back)
    img_back_blur = np.abs(img_back_blur)
    img_blur = np.uint8(img_back_blur)


    return img_blur

# Using Fast Fourier Transform to perform Edge Detection operation using Circular Mask
def detect_edges(thresh):

    fft_thresh_img = scipy.fft.fft2(thresh, axes = (0,1))
    fft_thresh_img_shifted = scipy.fft.fftshift(fft_thresh_img)
    
    radius = 125
    rows, cols = thresh.shape
    center_x, center_y = int(rows / 2), int(cols / 2)
    x, y = np.ogrid[:rows, :cols]
    Cmask_area = (x - center_x) ** 2 + (y - center_y) ** 2 <= np.square(radius)
    Cmask = np.ones((rows, cols)) 
    Cmask[Cmask_area] = 0
    
    fft_edge_img = fft_thresh_img_shifted * Cmask

    edge_img_back_shifted = scipy.fft.ifftshift(fft_edge_img)
    img_back_edge = scipy.fft.ifft2(edge_img_back_shifted)
    img_back_edge = np.abs(img_back_edge)

    return img_back_edge

## Detect the Corners
def features(image_gray,image):

    kernel = np.ones((11,11),np.uint8)
    erosion = cv2.erode(image_gray,kernel,iterations = 1)
    # image_dilated = cv2.dilate(erosion,kernel,iterations = 1)
    
    dst = cv2.cornerHarris(erosion,3,3,0.05)
    dst = cv2.dilate(dst,None)
    ret, dst = cv2.threshold(dst,0.01*dst.max(),255,0)
    dst = np.uint8(dst)

    # find centroids
    ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)

    # define the criteria to stop and refine the corners
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    corners = cv2.cornerSubPix(image_gray,np.float32(centroids),(5,5),(-1,-1),criteria)

    # print(corners)
    if len(corners) > 8:

        '''for i in corners:
            x_cord, y_cord = i.ravel()
            print(x_cord)
            cv2.circle(image, (x_cord, y_cord), 3, 255, -1)
            cv2.putText(image, "({},{})".format(x_cord, y_cord), (int(x_cord - 50), int(y_cord - 10) - 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1)'''
            
        x = []
        y = []

        for i in range(0,len(corners)):
            a = corners[i]
            x.append(int(a[0]))
            y.append(int(a[1]))

        Xmin_index = x.index(min(x))
        Xmin = x.pop(Xmin_index)
        Xmin_y = y.pop(Xmin_index)

        Xmax_index = x.index(max(x))
        Xmax = x.pop(Xmax_index)
        Xmax_y = y.pop(Xmax_index)

        Ymin_index = y.index(min(y))
        Ymin = y.pop(Ymin_index)
        Ymin_x = x.pop(Ymin_index)

        Ymax_index = y.index(max(y))
        Ymax = y.pop(Ymax_index)
        Ymax_x = x.pop(Ymax_index)


        image = cv2.line(image,(Xmin,Xmin_y),(Ymin_x,Ymin),(0,0,255),2)
        image = cv2.line(image,(Xmin,Xmin_y),(Ymax_x,Ymax),(0,0,255),2)
        image = cv2.line(image,(Ymax_x,Ymax),(Xmax,Xmax_y,),(0,0,255),2)
        image = cv2.line(image,(Ymin_x,Ymin),(Xmax,Xmax_y),(0,0,255),2)

        Xmin_index = x.index(min(x))
        Xmin = x.pop(Xmin_index)
        Xmin_y = y.pop(Xmin_index)

        Xmax_index = x.index(max(x))
        Xmax = x.pop(Xmax_index)
        Xmax_y = y.pop(Xmax_index)

        Ymin_index = y.index(min(y))
        Ymin = y.pop(Ymin_index)
        Ymin_x = x.pop(Ymin_index)

        Ymax_index = y.index(max(y))
        Ymax = y.pop(Ymax_index)
        Ymax_x = x.pop(Ymax_index)


        image = cv2.line(image,(Xmin,Xmin_y),(Ymin_x,Ymin),(0,0,255),2)
        image = cv2.line(image,(Xmin,Xmin_y),(Ymax_x,Ymax),(0,0,255),2)
        image = cv2.line(image,(Ymax_x,Ymax),(Xmax,Xmax_y,),(0,0,255),2)
        image = cv2.line(image,(Ymin_x,Ymin),(Xmax,Xmax_y),(0,0,255),2)

        corner_points = np.array(([Ymin_x,Ymin],[Xmin,Xmin_y],[Ymax_x,Ymax],[Xmax,Xmax_y]))

        desired_tag_corner = np.array([ [0, tag_size-1], [tag_size-1, tag_size-1], [tag_size-1, 0], [0, 0]])

        return image, corner_points, desired_tag_corner, Ymin, Ymax, Xmin, Xmax

    return image, None, None, None, None, None, None

## Function to perform Homography
def homography_matrix(corners1, corners2):

    if (len(corners1) < 4) or (len(corners2) < 4):
        print("Need atleast four points to compute SVD.")
        return 0

    x = corners1[:, 0]
    y = corners1[:, 1]
    xp = corners2[:, 0]
    yp = corners2[:,1]

    nrows = 8
    #ncols = 9
    
    A = []
    for i in range(int(nrows/2)):
        row1 = np.array([-x[i], -y[i], -1, 0, 0, 0, x[i]*xp[i], y[i]*xp[i], xp[i]])
        A.append(row1)
        row2 = np.array([0, 0, 0, -x[i], -y[i], -1, x[i]*yp[i], y[i]*yp[i], yp[i]])
        A.append(row2)

    A = np.array(A)
    U, E, VT = np.linalg.svd(A)
    V = VT.transpose()
    H_vertical = V[:, V.shape[1] - 1]
    H = H_vertical.reshape([3,3])
    H = H / H[2,2]
    # print("the Homography matrix is")
    # print(H)

    return H

## Warping function to get the TAG
def warpPerspective(H,img,maxHeight,maxWidth):
    H_inv=np.linalg.inv(H)
    warped=np.zeros((maxHeight,maxWidth,3),np.uint8)
    for i in range(maxHeight):
        for j in range(maxWidth):
            f = [i,j,1]
            f = np.reshape(f,(3,1))
            x, y, z = np.matmul(H_inv,f)
            xb = np.clip(x/z,0,1919)
            yb = np.clip(y/z,0,1079)
            # x, y, z = np.matmul(H,f)
            warped[i][j] = img[int(yb)][int(xb)]
    return(warped)


## Function for Extracting the inner grid of the Tag
def processreftag(ref_tag_image):
    tag_size = 160
    ref_tag_image_gray = cv2.cvtColor(ref_tag_image, cv2.COLOR_BGR2GRAY)
    ref_tag_image_thresh = cv2.threshold(ref_tag_image_gray, 230 ,255,cv2.THRESH_BINARY)[1]
    ref_image_thresh_resized = cv2.resize(ref_tag_image_thresh, (tag_size, tag_size))
    grid_size = 8
    stride = int(tag_size/grid_size)
    grid = np.zeros((8,8))
    x = 0
    y = 0
    for i in range(0, grid_size, 1):
        for j in range(0, grid_size, 1):
            cell = ref_image_thresh_resized[y:y+stride, x:x+stride]
            if cell.mean() > 255//2:
                grid[i][j] = 255
            x = x + stride
        x = 0
        y = y + stride
    inner_grid = grid[2:6, 2:6]
    return inner_grid

## Function to get the data from the inner grid of the TAG
def getinfotag(inner_grid):
    count = 0
    while not inner_grid[3,3] and count<4 :
        inner_grid = np.rot90(inner_grid,1)
        count+=1

    info_grid = inner_grid[1:3,1:3]
    info_grid_array = np.array((info_grid[0,0],info_grid[0,1],info_grid[1,1],info_grid[1,0]))
    tag_id = 0
    tag_id_bin = []
    for i in range(0,4):
        if(info_grid_array[i]) :
            tag_id = tag_id + 2**(i)
            tag_id_bin.append(1)
        else:
            tag_id_bin.append(0)

    return tag_id, tag_id_bin,count

## Function to super Impose the Image onto the TAG
def Testudo(image,testudo_img,corner_points,desired_tag_corner,Ymin,Ymax,Xmin,Xmax):

    rows,cols,ch = image.shape
    H = homography_matrix( np.float32(corner_points),np.float32(desired_tag_corner))
    
    h_inv = np.linalg.inv(H)

    for a in range(0,tag.shape[1]):
        for b in range(0,tag.shape[0]):
            x, y, z = np.matmul(h_inv,[a,b,1])
            xb = np.clip(x/z,0,1919)
            yb = np.clip(y/z,0,1079)
            image[int(yb)][int(xb)] = bilinear_interpolate(testudo_img, b, a)
    
    return image

## Rotate the Matrix until it is in Up-right Position
def rotatePoints(points):
    point_list = list(points.copy())
    top = point_list.pop(-1)
    point_list.insert(0, top)
    return np.array(point_list)

## Scaling the Image
def rescaleimage(frame,scale = 0.5):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimensions = (width,height)
    
    return cv2.resize(frame, dimensions, interpolation = cv2.INTER_AREA)

## Function to perform bilinear interpolation, using which the missing points in the superimposed image are filled.
def bilinear_interpolate(im, x, y):
    x = np.asarray(x)
    y = np.asarray(y)

    x0 = np.floor(x).astype(int)
    x1 = x0 + 1
    y0 = np.floor(y).astype(int)
    y1 = y0 + 1

    x0 = np.clip(x0, 0, im.shape[1]-1)
    x1 = np.clip(x1, 0, im.shape[1]-1)
    y0 = np.clip(y0, 0, im.shape[0]-1)
    y1 = np.clip(y1, 0, im.shape[0]-1)

    Ia = im[ y0, x0 ]
    Ib = im[ y1, x0 ]
    Ic = im[ y0, x1 ]
    Id = im[ y1, x1 ]

    wa = (x1-x) * (y1-y)
    wb = (x1-x) * (y-y0)
    wc = (x-x0) * (y1-y)
    wd = (x-x0) * (y-y0)

    return wa*Ia + wb*Ib + wc*Ic + wd*Id

## Function to calculate the Projection Matrix to Augment Cube on the TAG
def projectionMatrix(h, K):  
    h1 = h[:,0]          #taking column vectors h1,h2 and h3
    h2 = h[:,1]
    #h3 = h[:,2]

    #calculating lamda
    lamda = 2 / (np.linalg.norm(np.matmul(np.linalg.inv(K),h1)) + np.linalg.norm(np.matmul(np.linalg.inv(K),h2)))
    b_t = lamda * np.matmul(np.linalg.inv(K),h)

    #check if determinant is greater than 0 ie. has a positive determinant when object is in front of camera
    det = np.linalg.det(b_t)

    if det > 0:
        b = b_t
    else:                    #else make it positive
        b = -1 * b_t  
        
    row1 = b[:, 0]
    row2 = b[:, 1]                      #extract rotation and translation vectors
    row3 = np.cross(row1, row2)
    
    t = b[:, 2]
    Rt = np.column_stack((row1, row2, row3, t))

    P = np.matmul(K,Rt)  
    return(P,Rt,t)


# Main Function to Execute the Program
if __name__ == '__main__':
    
    ## Importing the Image and Video Files
    testudo_img = cv2.imread('testudo.png')

    # Defining the AR Tag size
    tag_size = 160

    testudo_img = cv2.resize(testudo_img, (tag_size,tag_size)) #Resize to the Tag Size

    ## Given Intrinsic Matrix Parameters
    K = np.array([[1346.1005953,0,932.163397529403],
       [ 0, 1355.93313621175,654.898679624155],
       [ 0, 0,1]])


    cap = cv2.VideoCapture('1tagvideo.mp4')

    ## Start getting Video Stream
    while(True):
        ret, frame = cap.read()
        if not ret:
            print("Streaming Stopped")
            break
        
        # Making to copy to preserve original Frame
        image = frame.copy()
        
        # Converting to Gray Scale
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Blur the Image
        image_blur = blur_img(image_gray)
        # Convert to Binary Image
        ret,thresh = cv2.threshold(image_blur, 220 ,255,cv2.THRESH_BINARY)
        # Getting the Detected Edges
        image_edge = detect_edges(thresh)
        # Converting into Image Data type
        image_edge = np.uint8(image_edge)
        
        # Obtain Corners
        frame,corner_points,desired_tag_corner,Ymin,Ymax,Xmin,Xmax = features(image_gray,image)
     
        # Loop until all the corner points are explored
        if corner_points is not None:
            
            # Homography and warp to get the Tag
            H = homography_matrix( np.float32(corner_points),np.float32(desired_tag_corner))
            tag = warpPerspective( H, image,tag_size, tag_size)
            
            # Getting Tag pose and data
            tag = cv2.cvtColor(np.uint8(tag), cv2.COLOR_BGR2GRAY)
            ret, tag = cv2.threshold(np.uint8(tag), 230 ,255,cv2.THRESH_BINARY)
            tag = cv2.cvtColor(tag,cv2.COLOR_GRAY2RGB)
            inner_grid = processreftag(tag) # returns the 2 x 2 inner grid
            tag_id, tag_id_bin, track_rotations = getinfotag(inner_grid) # extract data from inner grid

            # Track of rotations to get the TAG up-right
            for i in range(track_rotations):
                desired_tag_corner = rotatePoints(desired_tag_corner)
            
            # Superimposition of Testudo Image
            image = Testudo(image,testudo_img,corner_points,desired_tag_corner,Ymin,Ymax,Xmin,Xmax)
            
            # Recompute the Homography Matrix
            H = homography_matrix( np.float32(desired_tag_corner),np.float32(corner_points))
            
            # Projecting Virtual Cube onto Tag
            P,Rt,t = projectionMatrix(H,K)
            x1,y1,z1 = np.matmul(P,[0,0,0,1])
            x2,y2,z2 = np.matmul(P,[0,159,0,1])
            x3,y3,z3 = np.matmul(P,[159,0,0,1])
            x4,y4,z4 = np.matmul(P,[159,159,0,1])
            x5,y5,z5 = np.matmul(P,[0,0,-159,1])
            x6,y6,z6 = np.matmul(P,[0,159,-159,1])
            x7,y7,z7 = np.matmul(P,[159,0,-159,1])
            x8,y8,z8 = np.matmul(P,[159,159,-159,1])

            # Drawing the Lines of Cube
            cv2.line(image,(int(x1/z1),int(y1/z1)),(int(x5/z5),int(y5/z5)), (0,0,255), 2)
            cv2.line(image,(int(x2/z2),int(y2/z2)),(int(x6/z6),int(y6/z6)), (0,0,255), 2)
            cv2.line(image,(int(x3/z3),int(y3/z3)),(int(x7/z7),int(y7/z7)), (0,0,255), 2)
            cv2.line(image,(int(x4/z4),int(y4/z4)),(int(x8/z8),int(y8/z8)), (0,0,255), 2)

            cv2.line(image,(int(x1/z1),int(y1/z1)),(int(x2/z2),int(y2/z2)), (0,255,0), 2)
            cv2.line(image,(int(x1/z1),int(y1/z1)),(int(x3/z3),int(y3/z3)), (0,255,0), 2)
            cv2.line(image,(int(x2/z2),int(y2/z2)),(int(x4/z4),int(y4/z4)), (0,255,0), 2)
            cv2.line(image,(int(x3/z3),int(y3/z3)),(int(x4/z4),int(y4/z4)), (0,255,0), 2)

            cv2.line(image,(int(x5/z5),int(y5/z5)),(int(x6/z6),int(y6/z6)), (255,0,0), 2)
            cv2.line(image,(int(x5/z5),int(y5/z5)),(int(x7/z7),int(y7/z7)), (255,0,0), 2)
            cv2.line(image,(int(x6/z6),int(y6/z6)),(int(x8/z8),int(y8/z8)), (255,0,0), 2)
            cv2.line(image,(int(x7/z7),int(y7/z7)),(int(x8/z8),int(y8/z8)), (255,0,0), 2)

        try:
            cv2.imshow('Output Visualization',image)
        except:
            pass

        if cv2.waitKey(1) & 0xFF == ord('d'):
            break

    cap.release()
    cv2.destroyAllWindows()


