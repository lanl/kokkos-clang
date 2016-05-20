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

#include "clang/Analysis/CFG.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/ExternalASTSource.h"
#include "clang/AST/Stmt.h"
#include "clang/Parse/ParseDiagnostic.h"
#include "clang/Parse/Parser.h"
#include "clang/Sema/CodeCompleteConsumer.h"
#include "clang/Sema/ExternalSemaSource.h"
#include "clang/Sema/Sema.h"
#include "clang/Sema/SemaConsumer.h"
#include "llvm/Support/CrashRecoveryContext.h"
#include <cstdio>
#include <memory>

#include <unordered_set>
#include <unordered_map>

#include <iostream>

#include "clang/CodeGen/ideas/ASTVisitors.h"
#include "clang/CodeGen/ideas/ASTVisitors.h"

 #define ndump(X) llvm::errs() << __FILE__ << ":" << __LINE__ << ": " << \
 __PRETTY_FUNCTION__ << ": " << #X << " = " << X << "\n"

namespace clang{

  class ParallelAnalysis{
  public:
    using VarSet = std::set<const VarDecl*>;

    using DataMap = std::unordered_map<const Stmt*, VarSet>;

    struct Data{
      VarSet readViewVars;
      VarSet writeViewVars;
      VarSet readArrayVars;
      VarSet writeArrayVars;
    };

    typedef std::unordered_set<CFGBlock*> BlockSet;
    
    static void Run(Sema& S, FunctionDecl* fd){
      std::unique_ptr<CFG> cfg =
      CFG::buildCFG(fd, fd->getBody(), &S.Context, CFG::BuildOptions());

      BlockSet vs;
      Data data;
      Visit(cfg.get(), nullptr, vs, data, cfg->getEntry());
    }
                      
    static void Visit(CFG* cfg,
                      const Stmt* prevStmt,
                      BlockSet& vs,
                      Data& data,
                      CFGBlock& block){

      vs.insert(&block);

      const Stmt* lastStmt = prevStmt;

      for(auto itr = block.begin(), itrEnd = block.end();
          itr != itrEnd; ++itr){
        
        auto o = itr->getAs<CFGStmt>();
        if(!o.hasValue()){
          continue;
        }

        const Stmt* stmt = o.getValue().getStmt();

        CodeGen::PTXParallelConstructVisitor visitor(nullptr);

        bool isParallelConstruct = false;

        if(const CallExpr* ce = dyn_cast<CallExpr>(stmt)){

          const FunctionDecl* f = ce->getDirectCallee();

          if(f && (f->getQualifiedNameAsString() == "Kokkos::parallel_for" ||
                   f->getQualifiedNameAsString() == "Kokkos::parallel_reduce")){

            isParallelConstruct = true;

            visitor.Visit(const_cast<Stmt*>(stmt));
            
            VarSet remove;

            for(auto vd : data.writeViewVars){
              if(visitor.readViewVars().find(vd) != visitor.readViewVars().end()){                
                addVar(toDeviceViews_, lastStmt, vd);
                remove.insert(vd);
              }
            }

            for(auto vd : remove){
              data.writeViewVars.erase(vd);
            }

            remove.clear();

            for(auto vd : data.writeArrayVars){
              if(visitor.readArrayVars().find(vd) !=
                 visitor.readArrayVars().end()){
                addVar(toDeviceArrays_, lastStmt, vd);
                remove.insert(vd);
              }
            }

            for(auto vd : remove){
              data.writeArrayVars.erase(vd);
            }

            data.readViewVars.insert(visitor.writeViewVars().begin(),
              visitor.writeViewVars().end());

            data.readArrayVars.insert(visitor.writeArrayVars().begin(),
              visitor.writeArrayVars().end());
          }
        }

        if(!isParallelConstruct){
          visitor.Visit(const_cast<Stmt*>(stmt));

          bool found = !(visitor.writeViewVars().empty() &&
            visitor.writeArrayVars().empty());

          data.writeViewVars.insert(visitor.writeViewVars().begin(),
            visitor.writeViewVars().end());

          data.writeArrayVars.insert(visitor.writeArrayVars().begin(),
            visitor.writeArrayVars().end());

          VarSet remove;

          for(auto vd : visitor.readViewVars()){
            found = true;
            if(data.readViewVars.find(vd) != data.readViewVars.end()){
              addVar(fromDeviceViews_, stmt, vd);
              remove.insert(vd);
            }
          }

          for(auto vd : remove){
            data.readViewVars.erase(vd);
          }

          remove.clear();

          for(auto vd : visitor.readArrayVars()){
            found = true;
            if(data.readArrayVars.find(vd) != data.readArrayVars.end()){
              addVar(fromDeviceArrays_, stmt, vd);
              remove.insert(vd);
            }
          }

          for(auto vd : remove){
            data.readArrayVars.erase(vd);
          }

          if(found){
            lastStmt = stmt;
          }
        }
      }

      bool found = false;
      
      for(auto itr = block.succ_begin(), itrEnd = block.succ_end();
          itr != itrEnd; ++itr){
        CFGBlock::AdjacentBlock b = *itr;
        
        CFGBlock* block = b.getReachableBlock();
        
        if(block && vs.find(block) == vs.end()){
          Visit(cfg, lastStmt, vs, data, *block);
          found = true;
        }
      }
    }

    static void addVar(DataMap& m, const Stmt* s, const VarDecl* vd){
      auto itr = m.find(s);
      
      if(itr != m.end()){
        itr->second.insert(vd);
        return;  
      }

      VarSet vs;
      vs.insert(vd);
      m.emplace(s, std::move(vs));
    }

    static DataMap& toDeviceViews(){
      return toDeviceViews_;
    }

    static DataMap& fromDeviceViews(){
      return fromDeviceViews_;
    }

    static DataMap& toDeviceArrays(){
      return toDeviceArrays_;
    }

    static DataMap& fromDeviceArrays(){
      return fromDeviceArrays_;
    }

  private:
    static DataMap toDeviceViews_;
    static DataMap fromDeviceViews_;
    static DataMap toDeviceArrays_;
    static DataMap fromDeviceArrays_;
  };
  
} // end namespace clang
