# crnn

The Pytorch implementation is [meijieru/crnn.pytorch](https://github.com/meijieru/crnn.pytorch).

## How to wrap
<기본 로직>
Python ------호출-------> Library.so 
* Library.so 파일 안에는 C언어로 컴파일 가능한 함수가 담겨있어야 합니다.
→  C++ 함수를 사용하고 싶을 경우 extern "C"로 C++ 언어를 wrapping 하여 사용해야합니다.

* Python 코드 안에는 C언어로 작성된 함수의 argument와 return value의 자료형을 Python 자료형으로 변환해서 선언해주는 부분이 필요합니다.

<step 1> C++ 파일 수정
→  vi [파일명].h
→  vi [파일명].cpp

* 헤더파일의 생성은 자유입니다.

* Wrapper 코드(extern "C"로 wrapping 한 부분)는 하나의 .cpp파일에 작성해도 되고 편의상 따로 .cpp파일을 작성하셔도 무방합니다.

* Python에서 호출되지 않는 함수는 wrapping 하지 않아도 됩니다.

*이미지 처리 과정
1. <Python에서 이미지 load 및 이미지 데이터 정보 추출>
2. <Python에서 save_data 함수 호출>
3. <C++에서 정의된 save_data 함수 내에서 readfrombuffer 함수 호출>
4. <CV의 Mat 자료형으로 이미지 복원 및 저장>

<step 2>. C++ CMake로 .o 실행파일 .so 라이브러리 파일 생성 
   
→ .so 라이브러리 파일 구성하기

<step 3>. 파이썬 파일 작성

→  반환 값의 타입을 정의 : 라이브러리명.함수/객체명.restype = ctypes.c_자료형

→  인자 값의 타입을 정의 : 라이브러리명.함수/객체명.argtypes = [ctypes.c_자료형]

→  ctypes 자료형 정리 공식 홈페이지 참조 (https://docs.python.org/ko/3/library/ctypes.html)

→  반환 값이나 인자 값이 없을 시 예시로 void를 정의해주어야 segmentation fault 등이 발생하지 않습니다.

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

