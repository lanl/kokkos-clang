/*
 * ###########################################################################
 * Copyright (c) 2015, Los Alamos National Security, LLC.
 * All rights reserved.
 * 
 *  Copyright 2010. Los Alamos National Security, LLC. This software was
 *  produced under U.S. Government contract DE-AC52-06NA25396 for Los
 *  Alamos National Laboratory (LANL), which is operated by Los Alamos
 *  National Security, LLC for the U.S. Department of Energy. The
 *  U.S. Government has rights to use, reproduce, and distribute this
 *  software.  NEITHER THE GOVERNMENT NOR LOS ALAMOS NATIONAL SECURITY,
 *  LLC MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR ASSUMES ANY LIABILITY
 *  FOR THE USE OF THIS SOFTWARE.  If software is modified to produce
 *  derivative works, such modified software should be clearly marked,
 *  so as not to confuse it with the version available from LANL.
 *
 *  Additionally, redistribution and use in source and binary forms,
 *  with or without modification, are permitted provided that the
 *  following conditions are met:
 *
 *    * Redistributions of source code must retain the above copyright
 *      notice, this list of conditions and the following disclaimer.
 * 
 *    * Redistributions in binary form must reproduce the above
 *      copyright notice, this list of conditions and the following
 *      disclaimer in the documentation and/or other materials provided 
 *      with the distribution.
 *
 *    * Neither the name of Los Alamos National Security, LLC, Los
 *      Alamos National Laboratory, LANL, the U.S. Government, nor the
 *      names of its contributors may be used to endorse or promote
 *      products derived from this software without specific prior
 *      written permission.
 * 
 *  THIS SOFTWARE IS PROVIDED BY LOS ALAMOS NATIONAL SECURITY, LLC AND
 *  CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
 *  INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
 *  MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 *  DISCLAIMED. IN NO EVENT SHALL LOS ALAMOS NATIONAL SECURITY, LLC OR
 *  CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 *  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 *  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF
 *  USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 *  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 *  OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT
 *  OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 *  SUCH DAMAGE.
 * ########################################################################### 
 * 
 * Notes
 *
 * ##### 
 */

#ifndef LLVM_CLANG_LIB_CODEGEN_IDEAS_AST_VISITORS_H
#define LLVM_CLANG_LIB_CODEGEN_IDEAS_AST_VISITORS_H

#include <unordered_set>
#include <set>

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
    RHS
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
            default:
              break;
          }
        }
      }
      else if(const PointerType* pt = dyn_cast<PointerType>(ct.getTypePtr())){
        (void)pt;

        if(auto ne = dyn_cast<CXXNewExpr>(vd->getInit())){
          if(ne->isArray()){
            arrayVars_.insert(vd);
            switch(opType_){
              case OpType::LHS:
                writeArrayVars_.insert(vd);
                break;
              case OpType::RHS:
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
    if(S){
      for(Stmt::child_iterator I = S->child_begin(),
          E = S->child_end(); I != E; ++I){
        if(Stmt* child = *I){
          Visit(child);
        }
      }
    }
  }

  void VisitBinaryOperator(BinaryOperator* S){
    switch(S->getOpcode()){
    case BO_Assign:
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
      opType_ = OpType::LHS;
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
