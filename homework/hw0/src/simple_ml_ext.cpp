#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <iostream>

namespace py = pybind11;

template<typename type>
class Matrix {
private:
    type *pdata; // data pointer
    size_t rows;
    size_t cols;

public:
    Matrix(size_t rows, size_t cols, type *pdata) : pdata(pdata), rows(rows), cols(cols) {}

    ~Matrix() {}

    // Get number of rows
    size_t numRows() const {
        return rows;
    }

    // Get number of columns
    size_t numCols() const {
        return cols;
    }

    // Overload () operator to access elements
    type& operator()(size_t row, size_t col) {
        return pdata[cols * row + col];
    }

    void add(Matrix& other, Matrix& result) {
        if (rows != other.rows || cols != other.cols) {
            throw std::invalid_argument("Matrix dimensions must be the same for addition.");
        }

        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                result(i, j) = this->operator()(i, j) + other(i, j);
            }
        }
    }

    void sub(Matrix& other, Matrix& result) {
        if (rows != other.rows || cols != other.cols) {
            throw std::invalid_argument("Matrix dimensions must be the same for addition.");
        }

        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                result(i, j) = this->operator()(i, j) - other(i, j);
            }
        }
    }

    void dot(Matrix& other, Matrix& result) {
        if (cols != other.rows) {
            throw std::invalid_argument("Number of columns in the first matrix must be equal to the number of rows in the second matrix for multiplication.");
        }

        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < other.cols; ++j) {
                type sum = 0;
                for (size_t k = 0; k < cols; ++k) {
                    sum += this->operator()(i, k) * other(k, j);
                }
                result(i, j) = sum;
            }
        }
    }

    void T(Matrix& result) {
        for (size_t i = 0; i < cols; ++i) {
            for (size_t j = 0; j < rows; ++j) {
                result(i, j) = this->operator()(j, i);
            }
        }
    }

    // multiply scalar
    void mul(type m, Matrix& result) {
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                result(i, j) = this->operator()(i, j) * m;
            }
        }
    }

    void div(type d, Matrix& result) {
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                result(i, j) = this->operator()(i, j) / d;
            }
        }
    }

    void exp(Matrix& result) {
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                result(i, j) = std::exp(this->operator()(i, j));
            }
        }
    }

    void sum(Matrix& result) {
        for (size_t i = 0; i < rows; ++i) {
            type SUM = 0;
            for(size_t j = 0; j < cols; ++j) {
                SUM += this->operator()(i, j);
            }
            for(size_t j = 0; j < cols; ++j) {
                result(i, j) = SUM;
            }
        }
    }

    void elementwiseDiv(Matrix& other, Matrix& result) {
        if (rows != other.rows || cols != other.cols) {
            throw std::invalid_argument("Matrix dimensions must be the same for addition.");
        }

        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                result(i, j) = this->operator()(i, j) / other(i, j);
            }
        }
    }

    // Print matrix
    void print() const {
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                // type a = (*this)(i, j);
                std::cout << pdata[i*cols + j] << " ";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }
};

void softmax_regression_epoch_cpp(const float *X, const unsigned char *y,
								  float *theta, size_t m, size_t n, size_t k,
								  float lr, size_t batch)
{
    /**
     * A C++ version of the softmax regression epoch code.  This should run a
     * single epoch over the data defined by X and y (and sizes m,n,k), and
     * modify theta in place.  Your function will probably want to allocate
     * (and then delete) some helper arrays to store the logits and gradients.
     *
     * Args:
     *     X (const float *): pointer to X data, of size m*n, stored in row
     *          major (C) format
     *     y (const unsigned char *): pointer to y data, of size m
     *     theta (float *): pointer to theta data, of size n*k, stored in row
     *          major (C) format
     *     m (size_t): number of examples
     *     n (size_t): input dimension
     *     k (size_t): number of classes
     *     lr (float): learning rate / SGD step size
     *     batch (int): SGD minibatch size
     *
     * Returns:
     *     (None)
     */

    /// BEGIN YOUR CODE
    size_t num_examples = m;

    Matrix<float> Mtheta(n, k, theta);

    float *pscore = new float(batch*k);
    Matrix<float> score(batch, k, pscore);

    float *pexp_score = new float(batch*k);
    Matrix<float> exp_score(batch, k, pexp_score);

    float *pexp_score_sum = new float(batch*k);
    Matrix<float> exp_score_sum(batch, k, pexp_score_sum);

    float *pprobs = new float(batch*k);
    Matrix<float> probs(batch, k, pprobs);

    float *pX_batch_T = new float(n*batch);
    Matrix<float> X_batch_T(n, batch, pX_batch_T);

    float *pgrad = new float(n*k);
    Matrix<float> grad(n, k, pgrad);

    for(size_t i=0; i < num_examples; i+=batch) {
        float *p1 = (float *)X + n*i;
        const unsigned char *p2 = y + i;

        Matrix<float> X_batch(batch, n, p1);
        Matrix<const unsigned char> y_batch(batch, 1, p2);

        X_batch.dot(Mtheta, score);
        score.exp(exp_score);
//         exp_score.sum(exp_score_sum);
//         exp_score.elementwiseDiv(exp_score_sum, probs);
// 
//         for(size_t i=0; i< batch; ++i) {
//             probs(i, y_batch(i, 1)) -= 1;
//         }
// 
//         X_batch.T(X_batch_T);
//         X_batch_T.dot(probs, grad);
// 
//         grad.div(batch, grad);
//         grad.mul(lr, grad);
// 
//         Mtheta.sub(grad, Mtheta);
    }

    delete pscore;
    delete pexp_score;
    delete pexp_score_sum;
    delete pprobs;
    delete pX_batch_T;
    delete pgrad;
    /// END YOUR CODE
}


/**
 * This is the pybind11 code that wraps the function above.  It's only role is
 * wrap the function above in a Python module, and you do not need to make any
 * edits to the code
 */
PYBIND11_MODULE(simple_ml_ext, m) {
    m.def("softmax_regression_epoch_cpp",
    	[](py::array_t<float, py::array::c_style> X,
           py::array_t<unsigned char, py::array::c_style> y,
           py::array_t<float, py::array::c_style> theta,
           float lr,
           int batch) {
        softmax_regression_epoch_cpp(
        	static_cast<const float*>(X.request().ptr),
            static_cast<const unsigned char*>(y.request().ptr),
            static_cast<float*>(theta.request().ptr),
            X.request().shape[0],
            X.request().shape[1],
            theta.request().shape[1],
            lr,
            batch
           );
    },
    py::arg("X"), py::arg("y"), py::arg("theta"),
    py::arg("lr"), py::arg("batch"));
}
