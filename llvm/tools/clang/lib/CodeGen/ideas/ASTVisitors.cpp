#include "ASTVisitors.h"

using namespace clang;
using namespace CodeGen;

void ParallelForVisitor::VisitStmt(Stmt* S){
  bool shouldPop = false;
  
  if(CallExpr* ce = dyn_cast<CallExpr>(S)){
    const FunctionDecl* f = ce->getDirectCallee();
    if(f && (f->getQualifiedNameAsString() == "Kokkos::parallel_for" ||
             f->getQualifiedNameAsString() == "Kokkos::parallel_reduce")){
      ParallelForInfo* info = new ParallelForInfo;
      info->callExpr = ce;
      
      if(!stack_.empty()){
        ParallelForInfo* aboveInfo = stack_.back();
        aboveInfo->children.push_back(info);
        shouldPop = true;
      }
      
      stack_.push_back(info);
    }
  }
  
  VisitChildren(S);
  
  if(shouldPop){
    stack_.pop_back();
  } 
}
