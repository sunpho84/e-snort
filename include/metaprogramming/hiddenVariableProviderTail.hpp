/// \file hiddenVariableProviderTail.hpp
///
///  This file should be included at th end of each hidden variable
///  provider to reset for next ones. Please check that the macro
///  below agrees with that defined in the hiddenVariableProvider file


#ifdef DEFINE_HIDDEN_VARIABLES_ACCESSORS

# undef _HIDDENVARIABLEPROVIDER_HPP
# undef  DEFINE_OR_DECLARE_HIDDEN_VARIABLE_WITH_CONST_ACCESSOR

#endif
