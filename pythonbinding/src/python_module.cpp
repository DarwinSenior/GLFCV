#define PY_ARRAY_UNIQUE_SYMBOL pbcvt_ARRAY_API

#include <boost/python.hpp>
#include <boost/foreach.hpp>
#include <pyboostcvconverter/pyboostcvconverter.hpp>
#include <iostream>
#include "../../src/lf-depth-est.h"
#include "../../src/decoder.h"
#include "../../src/helper.h"

namespace GLFCV {

    using namespace boost::python;

    std::vector<std::vector<cv::Mat> > lf2mats(PyObject* lf) {
      std::vector<std::vector<cv::Mat> > mats(0);
      PyObject* it1 = PyObject_GetIter(lf);
      PyObject* item1;
      int i = 0;
      while (item1 = PyIter_Next(it1)) {
        PyObject* it2 = PyObject_GetIter(item1);
        PyObject* item2;
        std::vector<cv::Mat> mat2(0);
        while (item2 = PyIter_Next(it2)) {
          mat2.push_back(pbcvt::fromNDArrayToMat(item2));
          Py_DECREF(item2);
        }
        mats.push_back(mat2);
        Py_DECREF(it2);
        Py_DECREF(item1);
      }
      Py_DECREF(it1);
      return mats;
    }

    PyObject* disparity_map(
        PyObject* lf,
        float disp_min, float disp_max) {
      cv::Mat Dmat;
      auto lfmat = lf2mats(lf);
      BuildLFDisparityMap(lfmat, Dmat, disp_min, disp_max);
      auto ret = pbcvt::fromMatToNDArray(Dmat);
      return ret;
    }

#if (PY_VERSION_HEX >= 0x03000000)

    static void *init_ar() {
#else
        static void init_ar(){
#endif
        Py_Initialize();

        import_array();
        return NUMPY_IMPORT_ARRAY_RETVAL;
    }

    BOOST_PYTHON_MODULE (GLFCV) {
        //using namespace XM;
        init_ar();

        //initialize converters
        to_python_converter<cv::Mat,
                pbcvt::matToNDArrayBoostConverter>();
        pbcvt::matFromNDArrayBoostConverter();

        //expose module-level functions
        def("disparity_map", disparity_map);

    }

} //end namespace pbcvt
