# crnn

The Pytorch implementation is [meijieru/crnn.pytorch](https://github.com/meijieru/crnn.pytorch).

## How to wrap
<Basic Concept>
Python ------call-------> Library.so 
- Library.so file should include function which can be compiled with C language.
- If you want to use C++ function, you have to do wrapping C++ file with extern "C"
- Python code should include declaration of data types about argument and return value for C functions.

<step 1> Modifying C++ files 
- vi [file_name].h
- vi [file_name].cpp
- You can write header files separately.

<step 2>. Build up your .so libralies with CMake 
- g++ -c -fPIC [file_name].cpp -o [operation_file_name].o
- g++ -shared -Wl,-soname,[library_file_name].so -o [library_file_name].so [operation_file_name].o
- export LD_LIBRARY_PATH=[your_library_path]${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

<step 3>. Write python file
- define data type of return value
   - ex) library_file_name.function(or object...).restype = ctypes.c_datatype
- define data type of parameters
   - ex) library_file_name.function(or object...).argtypes = [ctypes.c_datatype]
- official link about ctypes data types
   - https://docs.python.org/ko/3/library/ctypes.html

## How to Run

```
1. generate crnn.wts from pytorch

git clone https://github.com/wang-xinyu/tensorrtx.git
git clone https://github.com/meijieru/crnn.pytorch.git
// download its weights 'crnn.pth'
// copy tensorrtx/crnn/genwts.py into crnn.pytorch/
// go to crnn.pytorch/
python genwts.py
// a file 'crnn.wts' will be generated.

2. build tensorrtx/crnn and run

// put crnn.wts into tensorrtx/crnn
// go to tensorrtx/crnn
mkdir build
cd build
cmake ..
make
sudo ./crnn -s  // serialize model to plan file i.e. 'crnn.engine'
// copy crnn.pytorch/data/demo.png here
sudo ./crnn -d  // deserialize plan file and run inference

3. check the output as follows:

raw: a-----v--a-i-l-a-bb-l-e---
sim: available

```

## More Information

See the readme in [home page.](https://github.com/wang-xinyu/tensorrtx)

## Acknowledgment

Thanks for the donation for this crnn tensorrt implementation from @雍.

