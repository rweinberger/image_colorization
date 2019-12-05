#include "matrix.h"
#include "Image.h"
#include "basicImageManipulation.h"
#include <cassert>
#include <ctime>
#include <iostream>

using namespace std;

float sumImage(Image im) {
  float total = 0.f;
  for (int i = 0; i < im.number_of_elements(); i++) {
    total += im(i);
  }
  return total;
}

Image colorize(Image bw, Image user_colored, float intensity_threshold, int size_window) {

  Image scribbles(bw.width(), bw.height(), 1);
  Image diff = bw - user_colored;
  for (int x = 0; x < scribbles.width(); x++) {
    for (int y = 0; y < scribbles.height(); y++) {
      float sum = abs(diff(x,y,0)) + abs(diff(x,y,1)) + abs(diff(x,y,2));
      scribbles(x,y) = sum > intensity_threshold ? 1 : 0;
    }
  }
  scribbles.debug_write(); // seems to be working ok

  Image bw_yuv = rgb2yuv(bw);
  Image user_yuv = rgb2yuv(user_colored);

  // populate output with 
  Image output_yuv(user_yuv.width(), user_yuv.height(), 3);
  for (int x = 0; x < user_yuv.width(); x++) {
    for (int y = 0; y < user_yuv.height(); y++) {
     output_yuv(x,y,0) = bw_yuv(x,y,0);
     output_yuv(x,y,1) = user_yuv(x,y,1);
     output_yuv(x,y,2) = user_yuv(x,y,2);
    }
  }
  output_yuv.debug_write();

  int m = bw.width();
  int n = bw.height();
  int next_index = -1;
  int pixel_index = -1;
  Array i_indices = Array::Zero(m * n * pow(2 * size_window + 1, 2));
  Array j_indices = Array::Zero(m * n * pow(2 * size_window + 1, 2));
  Array weight_values = Array::Zero(m * n * pow(2 * size_window + 1, 2));
  Array weights = Array::Zero(pow(2 * size_window + 1, 2));


 
  for (int j = 0; j < n; j++) {
    for (int i = 0; i < m; i++) {
      pixel_index += 1;
      if (scribbles(i,j) == 0) {

        // get neighbors
        int neighbor_index = -1;
        for (int neighbor_i = max(0, i-size_window); neighbor_i <= min(m-1,i+size_window); neighbor_i++) {
          for (int neighbor_j = max(0, j-size_window); neighbor_j <= min(n-1,j+size_window); neighbor_j++) {
            if (neighbor_i != i  || neighbor_j != j) {
              next_index += 1;
              neighbor_index += 1;
              i_indices(next_index) = pixel_index;
              j_indices(next_index) = neighbor_i*n + neighbor_j;
              weights(neighbor_index) = output_yuv(neighbor_i, neighbor_j, 0);
            }
          }
        }

      // get weights_values and i/j_indices
       float currentY = output_yuv(i,j,0);
       weights(neighbor_index + 1) = currentY; 

       Array subAry = weights.head(neighbor_index+1);
       float var = ((subAry - subAry.mean()).pow(2)).mean();
       float csig = 0.6*var;
       float min_weight = ((weights.head(neighbor_index) - currentY).pow(2)).minCoeff();
       csig = max(0.000002f, max(csig, -min_weight / float(log(0.01))));
       
       for (int k = 0; k <= neighbor_index; k++) {
        weights(k) = exp(-pow(weights(k) - currentY,2) / csig);
        weights(k) = weights(k) / weights.head(neighbor_index).sum();
       }
       int weight_index = 0;
       for (int k = next_index - neighbor_index + 1; k <= next_index; k++) {
        weight_values(k) = -weights(weight_index);
        weight_index += 1;
       }
      }
      next_index += 1;
      i_indices(next_index) = pixel_index;
      j_indices(next_index) = i*n + j;
      weight_values(next_index) = 1;
    }
  }

  // Getting rid of un-used spots in matricies
  weight_values = weight_values.head(next_index);
  i_indices = i_indices.head(next_index);
  j_indices = j_indices.head(next_index);

  // Make A and sparse A
  Matrix A(m*n, m*n);
  for (int i = 0; i < i_indices.rows(); i++) {
    A(i_indices(i), j_indices(i)) = weight_values(i);
  }
  Eigen::SparseMatrix<float> A_sparse = A.sparseView();

  Matrix b = Matrix::Zero(A_sparse.rows(), 1);

  for (int c = 1; c <= 2; ++c) {
    for (int x = 0; x < scribbles.width(); x++) {
      for (int y = 0; y < scribbles.height(); y++) {
        // if scribble is here, we want to take its chrominance
        if (scribbles(x,y) == 1) {
          b(x*n+y) = output_yuv(x,y,c);
        }
      }
    }

    // different ways to solve linear system
    Matrix answer = A.inverse() * b;
    // Eigen::PartialPivLU<Matrix> lu(A_sparse);
    // Matrix answer = lu.solve(b);
    // Matrix answer = (A.transpose() * A).ldlt().solve(A.transpose() * b);

    // put answer in output image
    for (int x = 0; x < output_yuv.width(); x++) {
      for (int y = 0; y < output_yuv.height(); y++) {
        output_yuv(x,y,c) = answer(x*n+y);
      }
    }
  }
  output_yuv.debug_write();

  return yuv2rgb(output_yuv);

}

// This is a way for you to test your functions.
int main() {

  // To run, just do make run in terminal


  Image bw("./Input/test_small_new.png");
  Image user_colored("./Input/test_small_color_new.png");

  cout << sumImage(user_colored) << endl;
  cout << sumImage(bw) << endl;

  Image out = colorize(bw, user_colored, .01f, 1);
  out.debug_write();
}

