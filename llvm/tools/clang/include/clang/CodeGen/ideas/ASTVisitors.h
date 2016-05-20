#ifndef LLVM_CLANG_LIB_CODEGEN_IDEAS_AST_VISITORS_H
#define LLVM_CLANG_LIB_CODEGEN_IDEAS_AST_VISITORS_H

#include <unordered_set>
#include <set>
#include <iostream>

#include "clang/AST/CharUnits.h"
#include "clang/AST/Decl.h"
#include "clang/AST/Stmt.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/ExprObjC.h"
#include "clang/AST/Type.h"
#include "clang/AST/DeclVisitor.h"
#include "clang/AST/StmtVisitor.h"
#include "clang/CodeGen/ideas/shared.h"

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

class PTXParallelConstructVisitor : public StmtVisitor<PTXParallelConstructVisitor> {
public: 

  enum class OpType{
    None,
    LHS,
    RHS,
    RW
  };

  using VarSet = std::set<const VarDecl*>;

  PTXParallelConstructVisitor(const VarDecl* reduceVar)
  : reduceVar_(reduceVar){
  
  }
    
  void VisitDeclRefExpr(DeclRefExpr* E){
    if(const VarDecl* vd = dyn_cast<VarDecl>(E->getDecl())){
      QualType ct = vd->getType().getCanonicalType();
      if(const RecordType* rt = dyn_cast<RecordType>(ct.getTypePtr())){
        (void)rt;
        if(ct.getAsString().find("class Kokkos::View") == 0){
          viewVars_.insert(vd);
          switch(opType_){
            case OpType::LHS:
              writeViewVars_.insert(vd);
              break;
            case OpType::RHS:
              readViewVars_.insert(vd);
              break;
            case OpType::RW:
              writeViewVars_.insert(vd);
              readViewVars_.insert(vd);
              break;
            default:
              break;
          }
        }
      }
      else if(const PointerType* pt = dyn_cast<PointerType>(ct.getTypePtr())){
        (void)pt;

        if(auto ne = dyn_cast_or_null<CXXNewExpr>(vd->getInit())){
          if(ne->isArray()){
            arrayVars_.insert(vd);
            switch(opType_){
              case OpType::LHS:
                writeArrayVars_.insert(vd);
                break;
              case OpType::RHS:
                readArrayVars_.insert(vd);
                break;
              case OpType::RW:
                writeArrayVars_.insert(vd);
                readArrayVars_.insert(vd);
                break;
              default:
                break;
            }
          }
        }
      }      
    } 
  }

  void VisitChildren(Stmt* S){
    if(!S){
      return;
    }

    for(Stmt::child_iterator I = S->child_begin(),
        E = S->child_end(); I != E; ++I){
      if(*I){
        Visit(*I);
      }
    }
  }

  void VisitCastExpr(CastExpr* E){
    VisitChildren(E);
  }

  void VisitCallExpr(CallExpr* E){
    VisitChildren(E);
  }
  
  void VisitDeclStmt(DeclStmt* S){
    const VarDecl* vd = dyn_cast_or_null<const VarDecl>(S->getSingleDecl());
    if(!vd){
      VisitChildren(S);
      return;
    }

    opType_ = OpType::RHS;
    Visit(const_cast<Expr*>(vd->getInit()));
    opType_ = OpType::None;
  }

  void VisitBinaryOperator(BinaryOperator* S){
    switch(S->getOpcode()){
    case BO_Assign:
      opType_ = OpType::LHS;
      break;
    case BO_MulAssign:
    case BO_DivAssign:
    case BO_RemAssign:
    case BO_AddAssign:
    case BO_SubAssign:
    case BO_ShlAssign:
    case BO_ShrAssign:
    case BO_AndAssign:
    case BO_XorAssign:
    case BO_OrAssign:
      opType_ = OpType::RW;
      break;
    default:
      opType_ = OpType::RHS;
      break;
    }

    if(reduceVar_){
      if(auto dr = dyn_cast<DeclRefExpr>(S->getLHS())){
        if(dr->getDecl() == reduceVar_){
          reduceOps_.insert(S);
        }
      }
    }

    Visit(S->getLHS());
    opType_ = OpType::RHS;
    Visit(S->getRHS());
    opType_ = OpType::None;
  }

  void VisitUnaryOperator(UnaryOperator* S){
    if(reduceVar_){
      switch(S->getOpcode()){
      case UO_PostInc:
      case UO_PostDec:
      case UO_PreInc:
      case UO_PreDec:
        if(auto dr = dyn_cast<DeclRefExpr>(S->getSubExpr())){
          if(dr->getDecl() == reduceVar_){
            reduceOps_.insert(S);
          }
        }
        break;
      default:
        break;
      }
    }
  }

  void VisitStmt(Stmt* S){
    VisitChildren(S);
  }

  const VarSet& viewVars() const{
    return viewVars_;
  }

  const VarSet& readViewVars() const{
    return readViewVars_;
  }

  const VarSet& writeViewVars() const{
    return writeViewVars_;
  }

  const VarSet& arrayVars() const{
    return arrayVars_;
  }

  const VarSet& readArrayVars() const{
    return readArrayVars_;
  }

  const VarSet& writeArrayVars() const{
    return writeArrayVars_;
  }

  const auto& reduceOps(){
    return reduceOps_;
  }

private:
  using ReduceOpSet = std::unordered_set<const Stmt*>;

  ReduceOpSet reduceOps_;

  const VarDecl* reduceVar_;
  VarSet viewVars_;
  VarSet readViewVars_;
  VarSet writeViewVars_;
  VarSet arrayVars_;
  VarSet readArrayVars_;
  VarSet writeArrayVars_;
  OpType opType_ = OpType::None;
};

} // end namespace CodeGen
} // end namespace clang

#endif // LLVM_CLANG_LIB_CODEGEN_IDEAS_AST_VISITORS_H
