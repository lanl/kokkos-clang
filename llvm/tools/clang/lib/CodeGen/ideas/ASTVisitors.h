#ifndef LLVM_CLANG_LIB_CODEGEN_IDEAS_AST_VISITORS_H
#define LLVM_CLANG_LIB_CODEGEN_IDEAS_AST_VISITORS_H

#include "CGBuilder.h"
#include "CGDebugInfo.h"
#include "CGValue.h"
#include "CodeGenModule.h"
#include "CodeGenFunction.h"
#include "clang/AST/CharUnits.h"
#include "clang/AST/Decl.h"
#include "clang/AST/Stmt.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/ExprObjC.h"
#include "clang/AST/Type.h"
#include "clang/AST/DeclVisitor.h"
#include "clang/AST/StmtVisitor.h"

namespace clang {
namespace CodeGen {

class ParallelForVisitor : public StmtVisitor<ParallelForVisitor> {
public:
  using ParallelForInfoVec = std::vector<ParallelForInfo*>;
  
  ParallelForVisitor(){}
  
  void VisitChildren(Stmt* S){
    if(S){
      for(Stmt::child_iterator I = S->child_begin(),
          E = S->child_end(); I != E; ++I){
        if(Stmt* child = *I){
          Visit(child);
        }
      }
    }
  }
  
  void VisitStmt(Stmt* S);
  
  ParallelForInfo* getParallelForInfo(){
    assert(!stack_.empty());
    
    return stack_[0];
  }
  
private:
  ParallelForInfoVec stack_;
};

} // end namespace CodeGen
} // end namespace clang

#endif // LLVM_CLANG_LIB_CODEGEN_IDEAS_AST_VISITORS_H
