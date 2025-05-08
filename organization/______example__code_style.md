# Code Style

## Naming convention
	snake_case for functions, clases and variables for consistency with the stl
	PascalCase for enums and enum structs
	Wahtever_Is_This_Case for multiword template parameters

## Class definitions
	member variables are preceded by m_
	static member variables are preceded by s_
	the order: using delcarations, constexpr declarations, public memeber functions, private memeber functions, class member variables

## Pointers
	*x to dereference instead of * x
	(T*)x to cast instead of (T *)x
	T* f() for return types instead of T *()
	- Prefer east const for references and pointers
	T const* const instead of const T *const (because if you read backwards it is correct. Constant ptr to constant int)
	T const* instead of const T * (because if you read backwards is correct, Ptr to constant int)
	T* const instead of T *const (because if you read backwards is correct, Constant ptr to int)

## References
	T& x instead of T & x
	- Prefer east const for references and pointers
	T const& instead of const T&
