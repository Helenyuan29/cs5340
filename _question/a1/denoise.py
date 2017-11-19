
import cv2
import numpy as np
from scipy.stats import norm

def denoise(im,J,sigma,burn_in=5,it_cnt=15):
    # perform type conversions and ensure input \in {1,-1}
    noisy = im.astype(np.float32)
    noisy[noisy==255]=1
    noisy[noisy==0]=-1
    psi_t = {}
    psi_t['1'] = norm.pdf(noisy,1,sigma) # $\psi_t(+1)
    psi_t['-1'] = norm.pdf(noisy,-1,sigma) # $\psi_t(-1)
    avg = np.zeros(noisy.shape)

    sample = noisy.copy()# start from current observation
    for i in range(it_cnt):
        # direct implementation of equation 3
        # note that we use convolution to perform summation for performance reasons
        k = np.array([[0,1,0],[1,0,1],[0,1,0]])
        nbr_sum = cv2.filter2D(sample,-1,k) # $\sum_{s\in nbr(t)}{x_s}$
        p1 = np.exp(J*nbr_sum) * psi_t['1']
        p0 = np.exp(-J*nbr_sum) * psi_t['-1']
        prob = p1/(p1+p0)

        sample_mask = np.less(np.random.random(noisy.shape),prob)
        sample[...]=-1
        sample[sample_mask]=1

        if i>=burn_in:
            avg += sample
    avg /= (it_cnt-burn_in)

    avg[avg>0]=255
    avg[avg<=0]=0
    return avg.astype(np.uint8)

def check_param():
    im = cv2.imread('3_noise.png')[:,:,0]
    im_orig = cv2.imread('3.png')[:,:,0]
    search_range = [.1,.2,.3,.4,.5,.7,.8,.85,.9,.95,1,1.1,1.2,1.3,1.5,1.8,2.1,2.5,3,3.5,4,5,6,100]
    best_acc = 0
    best_val = -1
    for sigma in  search_range:
        denoised=denoise(im,1,sigma,burn_in=5,it_cnt=105)
        diff=cv2.absdiff(im_orig,denoised)
        #cv2.imshow('im',im)
        #cv2.imshow('denoised',denoised)
        accuracy = 1-(diff.sum()/255/diff.shape[0]/diff.shape[1])
        cv2.imwrite('exp/%f_%f.png'%(accuracy,sigma),denoised)
        print 'acc =%f ,sigma =%f'%(accuracy,sigma)
        if accuracy > best_acc:
            best_acc = accuracy
            best_val = sigma
        #cv2.waitKey(0)
    print 'acc =%f ,sigma =%f'%(best_acc,best_val)

def produce_output():
    im_list = ['1_noise.png','2_noise.png','3_noise.png','4_noise.png']
    for i in im_list:
        im = cv2.imread(i)[:,:,0]
        denoised = denoise(im,1,1)
        cv2.imwrite('denoised/%s'%i,denoised)

#check_param()
produce_output()
