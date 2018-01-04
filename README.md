# Using **Face Landmarks** in Face Pose Predictor and Aligner

## Source

[[Paper]Rapid Object Detection using a Boosted Cascade of Simple Features - Paul Viola and Michael Jones - 2001](https://www.cs.cmu.edu/~efros/courses/LBMV07/Papers/viola-cvpr-01.pdf).

[[Book]Computer Vision: Algorithms and Applications - Richard Szeliski - Springer - 2010](http://szeliski.org/Book/)

[[Paper]Constrained Local Neural Fields for robust facial landmark detection in the wild - Tadas Baltrusaitis, Peter Robinson, and Louis-Philippe Morency - 2013](https://www.cl.cam.ac.uk/~tb346/pub/papers/iccv2013.pdf)

[[Paper]Facial Landmark Detection by Deep Multi-task Learning - Zhanpeng Zhang, Ping Luo, Chen Change Loy, and Xiaoou Tang - 2014](http://personal.ie.cuhk.edu.hk/~ccloy/files/eccv_2014_deepfacealign.pdf).

[[Paper]One Millisecond Face Alignment with an Ensemble of Regression Trees - Vahid Kazemi and Josephine Sullivan - Royal Institute of Technology, Stockholm, Sweden](http://www.csc.kth.se/~vahidk/papers/KazemiCVPR14.pdf).

[[Wiki]Affine Transformation](https://en.wikipedia.org/wiki/Affine_transformation).

[[Blog]Dlib's Real-Time Face Pose Estimation - Davis King - 2014](http://blog.dlib.net/2014/08/real-time-face-pose-estimation.html).

### Datasets and Pre-trained Models

[Labeled Face Parts in the Wild (LFPW) Dataset - Kriegman-Belhumeur Vision Technologies, LLC](https://neerajkumar.org/databases/lfpw/).

[Helen dataset](http://www.ifp.illinois.edu/~vuongle2/helen/).

![](./etc/dataset_sample.png)
 
### Libraries and Examples

[Dlib](http://dlib.net/) and its [Python binding](https://pypi.python.org/pypi/dlib).

[OpenFace](https://cmusatyalab.github.io/openface/).

[Adam Geitgey's Blog: Modern Face Recognition with Deep Learning - Finding, posing and projecting faces](https://medium.com/@ageitgey/machine-learning-is-fun-part-4-modern-face-recognition-with-deep-learning-c3cffc121d78).

## Details

### Vấn đề

![Mark](./mark.png)

Với con người, thật quá dễ dàng để nhận thấy rằng, cả hai khuôn mặt trong bức ảnh trên đều thuộc về một người.

Tuy nhiên, với máy tính, sẽ tự động cho rằng **hai khuôn mặt trên là hai con người khác nhau**.

Vì vậy, một tiền xử lý dữ liệu là cần thiết ([OpenFace's Docs](http://openface-api.readthedocs.io/en/latest/openface.html#openface-aligndlib-class)) để giải quyết vấn đề này và phải thực hiện ngay sau khi chúng ta đã cô lập được các khuôn mặt ra khỏi ảnh.

### Hướng tiếp cận

Để giải quyết vấn đề này, suy nghĩ đơn giản nhất là "đưa các bộ phận đặc trưng (mắt, mũi, miệng, ...) luôn nằm ở một vị trí cố định".

### Face's Landmarks and Pose Predictor

TODO

### 2D feature-based alignment


#### Parametric transformations

Phép biến đổi tham số (parametric transformation) sẽ tạo sự biến dạng trên toàn bộ bức ảnh, mà trong đó, kết quả của sự biến đổi phụ thuộc vào tập giá trị hữu hạn các tham số.

![Basic set of 2D planar transformations.](./etc/basic_transform.png)

Trong bức ảnh trên là một số ví dụ trực quan về các phép biến đổi tham số hai chiều thường được sử dụng.

```python
    # Example

    # image_f is input, image_g is excepted output and h is transform function
    def forward_mapping(image_f, h, image_g):
        width, height = image_f.shape[;;2]
        for x in range(0, width):
            for y in range(0, height): # For every pixels in the image f
                new_x, new_y = h(x, y) # Compute the new destination location
                image_g[new_x][new_y] = image_f[x][y] # Copy pixel from f to g

```
![](./etc/forward_mapping.png)

```python
    #Example

    # image_f is input, image_g is excepted output and h is transform function
    def inverse_wrapping(image_f, h, image_g):
        width, height = image_g.shape[;;2]
        for x in range(0, width):
            for y in range(0, height): # For every pixels in the image g
            source_x, source_y = h(x, y) # Compute the source location
            g[x][y] = f[source_x][source_y] # Resample and copy to image_g 

```
![](./etc/inverse_wrapping.png)

### Affine Transformation

![](./etc/transformations.png)