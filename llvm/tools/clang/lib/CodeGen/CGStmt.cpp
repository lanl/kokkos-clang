//===--- CGStmt.cpp - Emit LLVM Code from Statements ----------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This contains code to emit Stmt nodes as LLVM code.
//
//===----------------------------------------------------------------------===//

#include "CodeGenFunction.h"
#include "CGDebugInfo.h"
#include "CodeGenModule.h"
#include "TargetInfo.h"
#include "clang/AST/StmtVisitor.h"
#include "clang/Basic/Builtins.h"
#include "clang/Basic/PrettyStackTrace.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/Sema/LoopHint.h"
#include "clang/Sema/SemaDiagnostic.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/IR/CallSite.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/InlineAsm.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/MDBuilder.h"

// +====== ideas =========================
#include "ideas/CGIdeasRuntime.h"
#include "clang/CodeGen/ideas/ASTVisitors.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Target/TargetSubtargetInfo.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Verifier.h"
#include "clang/AST/ParentMap.h"
#include "llvm/Analysis/Passes.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Vectorize.h"

#include <unordered_set>

namespace llvm{
  extern ModulePass* createNVVMReflectPass(const StringMap<int>& Mapping);
}

// ======================================

using namespace clang;
using namespace CodeGen;

// +===== ideas ==============================

namespace{

const uint32_t REDUCE_KERNEL = 0b1;

} // namespace

void CodeGenFunction::EmitParallelFor(const CallExpr* E,
                                      ParallelForKind K){
  using namespace llvm;
  using namespace std;
  
  auto& B = Builder;
  auto& R = CGM.getIdeasRuntime();
  LLVMContext& C = getLLVMContext();
  
  typedef vector<Value*> ValueVec;
  typedef vector<llvm::Type*> TypeVec;

  ParallelForInfo* parent;
  ParallelForInfo* info;

  if(parallelForStack_.empty()){
    ParallelForVisitor visitor;
    visitor.VisitStmt(const_cast<CallExpr*>(E));
    parent = nullptr;
    info = visitor.getParallelForInfo();
  }
  else{
    parent = parallelForStack_.back();
    for(auto i : parent->children){
      if(i->callExpr == E){
        info = i;
        break;
      }
    }
    assert(info);
  }
  
  parallelForStack_.push_back(info);

  BasicBlock* startBlock = B.GetInsertBlock(); 
  BasicBlock::iterator startPoint = B.GetInsertPoint();
  Function* startFunc = startBlock->getParent(); 

  assert(E->getNumArgs() == 3);
  
  const Expr* n = E->getArg(0);
  
  const LambdaExpr* le = GetLambda(E->getArg(1));

  assert(le && "expected a lambda");
  
  CompoundStmt* body = le->getBody();
  CXXMethodDecl* md = le->getCallOperator();
  
  assert(md->getNumParams() == 1);
  
  ParmVarDecl* indexVar = md->getParamDecl(0);

  llvm::Type* indexType = ConvertType(indexVar->getType());

  Address indexPtr = 
  ideasAddr(B.CreateAlloca(indexType, nullptr, "index.ptr"));
  
  Value* end = EmitAnyExprToTemp(n).getScalarVal();
  end = B.CreateZExtOrTrunc(end, indexType);
  
  Value* zero = ConstantInt::get(indexType, 0);
  Value* one = ConstantInt::get(indexType, 1);
  
  B.CreateStore(zero, indexPtr);
  
  Value* synch = B.CreateCall(R.CreateSynchFunc(), end);

  BasicBlock* LoopBlock = createBasicBlock("parallel_for.loop", startFunc);

  B.CreateBr(LoopBlock);
  B.SetInsertPoint(LoopBlock);

  Value* index = B.CreateLoad(indexPtr);

  if(K == PFK_Serial){
    setAddrOfLocalVar(indexVar, indexPtr);
    EmitStmt(body);
  }
  else if(K == PFK_Threads){    
    llvm::Function* func =
    llvm::Function::Create(R.QueueFuncTy,
                           llvm::Function::ExternalLinkage,
                           "parallel_for",
                           &CGM.getModule());
    
    auto aitr = func->arg_begin();
    aitr->setName("args");
    Value* argsVoidPtr = aitr++;

    parallelForStack_.back()->arg = argsVoidPtr;
    
    BasicBlock* entry = createBasicBlock("entry", func);
    B.SetInsertPoint(entry);

    TypeVec fields = {VoidPtrTy, Int32Ty};
    StructType* argsType = StructType::create(C, fields, "struct.func_args");
    Value* argsPtr = B.CreateBitCast(argsVoidPtr, llvm::PointerType::get(argsType, 0));
    Address indexPtr = ideasAddr(B.CreateStructGEP(nullptr, argsPtr, 1));

    setAddrOfLocalVar(indexVar, indexPtr);
    
    EmitStmt(body);

    //BasicBlock* bodyBlock = B.GetInsertBlock();

    if(info->children.empty()){
      ValueVec args = {argsVoidPtr};
      B.CreateCall(R.FinishFuncFunc(), args);
      B.CreateRetVoid();
    }

    B.SetInsertPoint(LoopBlock);
        
    Value* funcPtr = B.CreateBitCast(func, VoidPtrTy);

    Value* priority = ConstantInt::get(Int32Ty, parallelForStack_.size());

    ValueVec args = {synch, funcPtr, index, priority};
    B.CreateCall(R.QueueFuncFunc(), args);
  }
  else{
    assert(false && "invalid parallel for kind");
  }
  
  Value* nextIndex = B.CreateAdd(index, one);
    
  B.CreateStore(nextIndex, indexPtr);
  
  Value* cond = B.CreateICmpULT(nextIndex, end);
    
  BasicBlock* ExitBlock = createBasicBlock("parallel_for.exit", startFunc);
  
  B.CreateCondBr(cond, LoopBlock, ExitBlock);
    
  B.SetInsertPoint(ExitBlock);

  parallelForStack_.pop_back();

  if(parent){
    Value* argsVoidPtr = parallelForStack_.back()->arg;

    ValueVec args = {argsVoidPtr};
    B.CreateCall(R.FinishFuncFunc(), args);
  }

  ValueVec args = {synch};
  B.CreateCall(R.AwaitSynchFunc(), args);

  if(parent){
    B.CreateRetVoid(); 
    B.SetInsertPoint(startBlock, startPoint);   
  }
}

const LambdaExpr* CodeGenFunction::GetLambda(const Expr* E){
  if(auto me = dyn_cast<MaterializeTemporaryExpr>(E)){
    E = me->GetTemporaryExpr();
  }
  
  if(const CastExpr* c = dyn_cast<CastExpr>(E)){
    E = c->getSubExpr();
  }

  if(const CXXBindTemporaryExpr* c = dyn_cast<CXXBindTemporaryExpr>(E)){
    E = c->getSubExpr();
  }

  return dyn_cast<LambdaExpr>(E);
}

namespace{

  enum class ReduceType{
    None,
    Sum,
    Product
  };

  llvm::Value* reduceOp(clang::CodeGen::CGBuilderTy& B,
                        ReduceType rt,
                        llvm::Value* a,
                        llvm::Value* b){
    
    if(a->getType()->isFloatingPointTy()){
      if(rt == ReduceType::Sum){
        return B.CreateFAdd(a, b);
      }
      else{
        return B.CreateFMul(a, b);
      }
    }
    else{
      if(rt == ReduceType::Sum){
        return B.CreateAdd(a, b);
      }
      else{
        return B.CreateMul(a, b);
      }
    }
  }

} // namespace

void CodeGenFunction::EmitParallelConstructPTX(const CallExpr* E){
  using namespace llvm;
  using namespace std;

  using ValueVec = vector<Value*>;
  using TypeVec = vector<llvm::Type*>;
  
  auto& B = Builder;
  auto& R = CGM.getIdeasRuntime();
  LLVMContext& C = getLLVMContext();

  const VarDecl* reduceVar;

  const LambdaExpr* le = GetLambda(E->getArg(1));

  assert(le && "expected a lambda");
  
  vector<const VarDecl*> captureVars;

  for(auto ci : le->captures()){
    bool shouldCapture = true;

    const VarDecl* vd = ci.getCapturedVar();

    QualType ct = vd->getType().getCanonicalType();

    if(const RecordType* rt = dyn_cast<RecordType>(ct.getTypePtr())){
      (void)rt;
      if(ct.getAsString().find("class Kokkos::View") == 0){
        shouldCapture = false;
      }
    }
    else if(const PointerType* pt = dyn_cast<PointerType>(ct.getTypePtr())){
      (void)pt;

      if(auto ne = dyn_cast<CXXNewExpr>(vd->getInit())){
        if(ne->isArray()){
          shouldCapture = false;
        }
      }
    }

    if(shouldCapture){
      captureVars.push_back(vd);
    }
  } 

  CXXMethodDecl* md = le->getCallOperator();

  const FunctionDecl* f = E->getDirectCallee();
  assert(f);

  if(f->getQualifiedNameAsString() == "Kokkos::parallel_for"){
    reduceVar = nullptr;
    assert(md->getNumParams() == 1);
    assert(E->getNumArgs() == 3);
  }
  else if(f->getQualifiedNameAsString() == "Kokkos::parallel_reduce"){
    reduceVar = md->getParamDecl(1);
    assert(md->getNumParams() == 2);
    assert(E->getNumArgs() == 5); 
  }
  else{
    assert(false && "expected parallel for or reduce");
  }

  ParallelForInfo* parent;
  ParallelForInfo* info;

  PTXParallelConstructVisitor::VarSet viewVars;
  PTXParallelConstructVisitor::VarSet readViewVars;
  PTXParallelConstructVisitor::VarSet writeViewVars;

  PTXParallelConstructVisitor::VarSet arrayVars;
  PTXParallelConstructVisitor::VarSet readArrayVars;
  PTXParallelConstructVisitor::VarSet writeArrayVars;

  ReduceType reduceType = ReduceType::None;

  if(parallelForStack_.empty()){
    ParallelForVisitor visitor;
    visitor.VisitStmt(const_cast<CallExpr*>(E));
    parent = nullptr;
    info = visitor.getParallelForInfo();

    PTXParallelConstructVisitor ptxVisitor(reduceVar);
    ptxVisitor.VisitStmt(const_cast<CallExpr*>(E));

    viewVars = ptxVisitor.viewVars();
    readViewVars = ptxVisitor.readViewVars();
    writeViewVars = ptxVisitor.writeViewVars();

    arrayVars = ptxVisitor.arrayVars();
    readArrayVars = ptxVisitor.readArrayVars();
    writeArrayVars = ptxVisitor.writeArrayVars();

    if(reduceVar){      
      for(auto op : ptxVisitor.reduceOps()){
        if(auto bo = dyn_cast<BinaryOperator>(op)){
          if(bo->getOpcode() == BO_AddAssign){
            
            assert(reduceType == ReduceType::None || 
                   reduceType == ReduceType::Sum);
            
            reduceType = ReduceType::Sum;
          }
          else if(bo->getOpcode() == BO_MulAssign){

            assert(reduceType == ReduceType::None || 
                   reduceType == ReduceType::Product);

            reduceType = ReduceType::Product;
          }
          else{
            assert(false && "invalid reduce type");
          }
        }
        else if(auto uo = dyn_cast<UnaryOperator>(op)){
          (void)uo;
          assert(reduceType == ReduceType::None || 
                 reduceType == ReduceType::Sum);
          
          reduceType = ReduceType::Sum;
        }
      }

      assert(reduceType != ReduceType::None && "failed to find reduce operator");
    }
  }
  else{
    parent = parallelForStack_.back();

    for(auto i : parent->children){
      if(i->callExpr == E){
        info = i;
        break;
      }
    }
    assert(info);
  }

  for(const VarDecl* vd : viewVars){    
    ViewInfo info;

    const RecordType* rt = 
      dyn_cast<RecordType>(vd->getType().getCanonicalType().getTypePtr());
    assert(rt && "expected a RecordType");

    const ClassTemplateSpecializationDecl* td = 
      dyn_cast<ClassTemplateSpecializationDecl>(rt->getDecl());
    assert(td && "expected a ClassTemplateSpecializationDecl");

    const TemplateArgumentList& tl = td->getTemplateArgs();
    TemplateArgument ta = tl[0];
    QualType tt = ta.getAsType();

    const PointerType* pt;
    const ConstantArrayType* at;

    for(;;){
      if((at = dyn_cast<ConstantArrayType>(tt.getTypePtr()))){
        info.staticSizes.push_back(at->getSize().getZExtValue());
        tt = at->getElementType();
      }
      else if((pt = dyn_cast<PointerType>(tt.getTypePtr()))){
        break;
      }
      else{
        assert(false && "expected a ConstantArrayType or PointerType");        
      }
    }

    info.runtimeDims = 1;

    for(;;){
      QualType et = pt->getPointeeType();

      pt = dyn_cast<PointerType>(et.getTypePtr());

      if(!pt){
        const BuiltinType* bt = dyn_cast<BuiltinType>(et.getTypePtr());
        assert(bt && "expected a BultinType");

        info.elementType = et;
        break;
      }

      ++info.runtimeDims;
    }

    viewInfoMap_[vd] = info;
  }

  for(const VarDecl* vd : arrayVars){    
    ArrayInfo info;

    const PointerType* pt = dyn_cast<PointerType>(vd->getType());
    assert(pt);

    info.elementType = pt->getPointeeType();

    auto ne = dyn_cast<CXXNewExpr>(vd->getInit());
    assert(ne);
    assert(ne->isArray());

    info.size = EmitAnyExprToTemp(ne->getArraySize()).getScalarVal();
    info.size = B.CreateTrunc(info.size, Int32Ty);

    arrayInfoMap_[vd] = info;
  }

  parallelForStack_.push_back(info);

  BasicBlock* startBlock = B.GetInsertBlock(); 
  BasicBlock::iterator startPoint = B.GetInsertPoint();
  Function* startFunc = startBlock->getParent(); 
    
  Value* nv = EmitScalarExpr(E->getArg(0));

  nv = B.CreateTrunc(nv, Int32Ty);

  CompoundStmt* body = le->getBody();
  
  ParmVarDecl* indexVar = md->getParamDecl(0);

  //llvm::Type* indexType = ConvertType(indexVar->getType());

  TypeVec params;

  for(const VarDecl* vd : viewVars){
    llvm::Type* t = ConvertType(viewInfoMap_[vd].elementType);
    params.push_back(llvm::PointerType::get(t, 0));
    params.push_back(llvm::PointerType::get(Int32Ty, 0));
  }

  for(const VarDecl* vd : arrayVars){
    const PointerType* pt = dyn_cast<PointerType>(vd->getType());
    assert(pt);

    llvm::Type* t = ConvertType(pt->getPointeeType());
    params.push_back(llvm::PointerType::get(t, 0));
  }

  for(const VarDecl* vd : captureVars){
    llvm::Type* t = ConvertType(vd->getType());
    params.push_back(t);
  }

  llvm::FunctionType* ft;
  llvm::FunctionType* ftReduce;

  if(reduceVar){
    llvm::Type* rt = ConvertType(reduceVar->getType().getNonReferenceType());
    params.push_back(Int32Ty);
    ft = llvm::FunctionType::get(rt, params, false);
    // shared memory
    //params.push_back(llvm::PointerType::get(rt, 0));
    params.push_back(ConvertType(reduceVar->getType()));
    ftReduce = llvm::FunctionType::get(VoidTy, params, false);
  }
  else{
    params.push_back(Int32Ty);
    ft = llvm::FunctionType::get(VoidTy, params, false);
  }

  llvm::Module ptxModule("PTXModule", C);
  CGIdeasRuntime KR(ptxModule);

  llvm::Function* func =
    llvm::Function::Create(ft,
                           llvm::Function::ExternalLinkage,
                           "run",
                           &ptxModule);

  //func->dump();

  auto aitr = func->arg_begin();

  parallelForParamMap_.clear();
  
  Value* reduceArray;

  for(const VarDecl* vd : viewVars){ 
    aitr->setName(vd->getName());
    parallelForParamMap_[vd] = aitr;
    ++aitr;

    string pn = vd->getName();
    pn += ".dims";
      
    aitr->setName(pn);

    parallelForParamDimMap_[vd] = aitr;
    ++aitr;
  }

  auto aitrArray = aitr;

  for(const VarDecl* vd : arrayVars){ 
    aitr->setName(vd->getName());
    ++aitr;
  }

  auto aitrCapture = aitr;

  for(const VarDecl* vd : captureVars){
    aitr->setName(vd->getName());
    ++aitr;
  }

  Value* count;
  if(reduceVar){
    aitr->setName("index");
  }
  else{
    count = aitr;
    aitr->setName("n");
    ++aitr;
  }

  llvm::Function* reduceFunc;

  Value* reduceIndex;
  Value* reduceN;
  Value* reduceShared;

  if(reduceVar){
    reduceIndex = aitr;

    reduceFunc =
      llvm::Function::Create(ftReduce,
                             llvm::Function::ExternalLinkage,
                             "reduce",
                             &ptxModule);

    auto aitr2 = reduceFunc->arg_begin();
    auto aitr3 = func->arg_begin();
    while(aitr3 != reduceIndex){
      aitr2->setName(aitr3->getName());
      ++aitr3;
      ++aitr2;
    }

    reduceN = aitr2;
    aitr2->setName("n");
    ++aitr2;
    reduceArray = aitr2;
    aitr2->setName("reduce.out");
  }
  else{
    reduceFunc = nullptr;
    reduceIndex = nullptr;
  } 

  llvm::Function* prevFn = CurFn;
  CurFn = func;

  BasicBlock* entry = createBasicBlock("entry", func);

  auto prevAllocaInsertPt = AllocaInsertPt;

  llvm::Value* Undef = llvm::UndefValue::get(Int32Ty);
  AllocaInsertPt = new llvm::BitCastInst(Undef, Int32Ty, "", entry);

  ReturnBlock = getJumpDestInCurrentScope("return");

  B.SetInsertPoint(entry);

  for(const VarDecl* vd : arrayVars){
    Value* varPtr = B.CreateAlloca(ConvertType(vd->getType()));
    B.CreateStore(aitrArray++, ideasAddr(varPtr));
    parallelForParamMap_[vd] = varPtr;
  }

  for(const VarDecl* vd : captureVars){
    Value* varPtr = B.CreateAlloca(ConvertType(vd->getType()));
    B.CreateStore(aitrCapture++, ideasAddr(varPtr));
    parallelForParamMap_[vd] = varPtr;
  }

  Value* reducePtr2;

  if(reduceVar){
    llvm::Type* pt = ConvertType(reduceVar->getType());
    llvm::Type* t = ConvertType(reduceVar->getType().getNonReferenceType());

    reduceShared = 
    new GlobalVariable(ptxModule, llvm::ArrayType::get(t, 4096), false,
                       llvm::GlobalValue::ExternalLinkage, nullptr, "shared",
                       nullptr, llvm::GlobalValue::NotThreadLocal, 3);

    reducePtr2 = B.CreateAlloca(pt);
    Value* ptr5 = B.CreateAlloca(t);
    B.CreateStore(ptr5, ideasAddr(reducePtr2));
    Value* indexPtr = B.CreateAlloca(Int32Ty);
    B.CreateStore(reduceIndex, ideasAddr(indexPtr)); 

    Value* initVal;
    
    if(t->isFloatingPointTy()){
      if(reduceType == ReduceType::Sum){
        initVal = ConstantFP::get(t, 0.0);
      }
      else if(reduceType == ReduceType::Product){
        initVal = ConstantFP::get(t, 1.0);
      }
      else{
        assert(false && "invalid reduce type");
      }
    }
    else{
      if(reduceType == ReduceType::Sum){
        initVal = ConstantInt::get(t, 0);
      }
      else if(reduceType == ReduceType::Product){
        initVal = ConstantInt::get(t, 1);
      }
      else{
        assert(false && "invalid reduce type");
      }
    }

    Value* reducePtr = B.CreateLoad(ideasAddr(reducePtr2));
    B.CreateStore(initVal, ideasAddr(reducePtr));
    setAddrOfLocalVar(reduceVar, ideasAddr(reducePtr2));

    setAddrOfLocalVar(indexVar, ideasAddr(indexPtr));  
  }
  else{
    Value* threadIdx = B.CreateCall(KR.getSREGFunc("tid.x"));
    Value* blockIdx = B.CreateCall(KR.getSREGFunc("ctaid.x"));
    Value* blockDim = B.CreateCall(KR.getSREGFunc("ntid.x"));

    Value* threadId = 
      B.CreateAdd(threadIdx, B.CreateMul(blockIdx, blockDim), "threadId");

    Value* cond = 
      B.CreateICmpUGT(threadId, count, "cond");

    BasicBlock* RetBlock = createBasicBlock("parallel_for.ret", func);

    BasicBlock* ContBlock = createBasicBlock("parallel_for.cont", func);

    B.CreateCondBr(cond, RetBlock, ContBlock);
    
    B.SetInsertPoint(RetBlock);

    B.CreateRetVoid();

    B.SetInsertPoint(ContBlock);

    Address threadIdPtr = 
    ideasAddr(B.CreateAlloca(threadId->getType(), nullptr, "threadId.ptr"));

    B.CreateStore(threadId, threadIdPtr);

    setAddrOfLocalVar(indexVar, threadIdPtr);
  }
    
  EmitStmt(body);

  if(reduceVar){
    Value* retPtr = B.CreateLoad(ideasAddr(reducePtr2));
    Value* ret = B.CreateLoad(ideasAddr(retPtr));
    B.CreateRet(ret);
  }
  else{
    B.CreateRetVoid();
  }

  //llvm::errs() << "---------------- kernel func\n";
  //func->dump();

  // =========================== REDUCE FUNCTION
  if(reduceVar){
    llvm::Type* rt = ConvertType(reduceVar->getType().getNonReferenceType());

    llvm::Function* barrierFunc = KR.getBarrierFunc();

    BasicBlock* entry = createBasicBlock("entry", reduceFunc);

    llvm::Value* Undef = llvm::UndefValue::get(Int32Ty);
    AllocaInsertPt = new llvm::BitCastInst(Undef, Int32Ty, "", entry);

    ReturnBlock = getJumpDestInCurrentScope("return");

    B.SetInsertPoint(entry);

    llvm::Type* pt = llvm::PointerType::get(rt, 3);

    reduceShared = B.CreateBitCast(reduceShared, pt);

    Value* threadId = B.CreateCall(KR.getSREGFunc("tid.x"));
    Value* blockIdx = B.CreateCall(KR.getSREGFunc("ctaid.x"));
    Value* gridDim = B.CreateCall(KR.getSREGFunc("nctaid.x"));

    Value* i512 = llvm::ConstantInt::get(Int32Ty, 512);
    Value* i256 = llvm::ConstantInt::get(Int32Ty, 256);
    Value* i128 = llvm::ConstantInt::get(Int32Ty, 128);
    Value* i64 = llvm::ConstantInt::get(Int32Ty, 64);
    Value* i32 = llvm::ConstantInt::get(Int32Ty, 32);
    Value* i16 = llvm::ConstantInt::get(Int32Ty, 16);
    Value* i8 = llvm::ConstantInt::get(Int32Ty, 8);
    Value* i4 = llvm::ConstantInt::get(Int32Ty, 4);
    Value* i2 = llvm::ConstantInt::get(Int32Ty, 2);
    Value* i1 = llvm::ConstantInt::get(Int32Ty, 1);
    Value* i0 = llvm::ConstantInt::get(Int32Ty, 0);

    Value* i = B.CreateAdd(B.CreateMul(blockIdx, i512), threadId, "i");
    Value* iPtr = B.CreateAlloca(i->getType());
    B.CreateStore(i, ideasAddr(iPtr));

    Value* gridSize = B.CreateMul(i512, gridDim);

    Value* initVal;

    if(rt->isFloatingPointTy()){
      if(reduceType == ReduceType::Sum){
        initVal = ConstantFP::get(rt, 0.0);
      }
      else if(reduceType == ReduceType::Product){
        initVal = ConstantFP::get(rt, 1.0);
      }
      else{
        assert(false && "invalid reduce type");
      }
    }
    else{
      if(reduceType == ReduceType::Sum){
        initVal = ConstantInt::get(rt, 0);
      }
      else if(reduceType == ReduceType::Product){
        initVal = ConstantInt::get(rt, 1);
      }
      else{
        assert(false && "invalid reduce type");
      }
    }

    Value* sPtr = B.CreateAlloca(rt);
    B.CreateStore(initVal, ideasAddr(sPtr));

    BasicBlock* CondBlock = createBasicBlock("cond", reduceFunc);

    B.CreateBr(CondBlock);

    B.SetInsertPoint(CondBlock);

    i = B.CreateLoad(ideasAddr(iPtr));

    Value* cond = B.CreateICmpULT(i, reduceN);

    BasicBlock* LoopBlock = createBasicBlock("loop", reduceFunc);
    BasicBlock* MergeBlock = createBasicBlock("merge", reduceFunc);

    B.CreateCondBr(cond, LoopBlock, MergeBlock);

    B.SetInsertPoint(LoopBlock);

    ValueVec args;

    size_t m = reduceFunc->arg_size() - 2;
    size_t j = 0;

    auto aitr2 = reduceFunc->arg_begin();
    for(auto& arg : reduceFunc->args()){
      if(j >= m){
        break;
      }
      
      args.push_back(aitr2);
      ++aitr2;
      ++j;
    }

    ValueVec args2 = args;
    args2.push_back(i);

    Value* val = B.CreateLoad(ideasAddr(sPtr));
    Value* ci = B.CreateCall(func, args2);
    //Value* ci = llvm::ConstantFP::get(rt, 1.0);
    val = reduceOp(B, reduceType, val, ci);
    B.CreateStore(val, ideasAddr(sPtr));

    BasicBlock* Cond2Block = createBasicBlock("cond2", reduceFunc);
    BasicBlock* Merge2Block = createBasicBlock("merge2", reduceFunc);

    cond = B.CreateICmpULT(B.CreateAdd(i, i256), reduceN);
    B.CreateCondBr(cond, Cond2Block, Merge2Block);

    B.SetInsertPoint(Cond2Block);

    val = B.CreateLoad(ideasAddr(sPtr));
    args2 = args;
    args2.push_back(B.CreateAdd(i, i256));
    ci = B.CreateCall(func, args2);
    //ci = llvm::ConstantFP::get(rt, 1.0);
    val = reduceOp(B, reduceType, val, ci);
    B.CreateStore(val, ideasAddr(sPtr));
    B.CreateBr(Merge2Block);

    B.SetInsertPoint(Merge2Block);
    
    i = B.CreateAdd(i, gridSize);

    B.CreateStore(i, ideasAddr(iPtr));

    B.CreateBr(CondBlock);

    B.SetInsertPoint(MergeBlock);

    Value* idx = B.CreateGEP(reduceShared, threadId);
    Address idxAddr = ideasAddr(idx);

    val = B.CreateLoad(ideasAddr(sPtr));

    B.CreateStore(val, idxAddr);

    B.CreateCall(barrierFunc);

    // cond 128

    BasicBlock* Cond3Block = createBasicBlock("cond3", reduceFunc);
    BasicBlock* Merge3Block = createBasicBlock("merge3", reduceFunc);

    cond = B.CreateICmpULT(threadId, i128);
    B.CreateCondBr(cond, Cond3Block, Merge3Block);
    B.SetInsertPoint(Cond3Block);

    val = B.CreateLoad(ideasAddr(sPtr));
    Value* idx128 = B.CreateGEP(reduceShared, B.CreateAdd(threadId, i128));
    Value* v128 = B.CreateLoad(ideasAddr(idx128));
    val = reduceOp(B, reduceType, val, v128);
    B.CreateStore(val, ideasAddr(sPtr));
    B.CreateStore(val, idxAddr);
    B.CreateBr(Merge3Block);

    B.SetInsertPoint(Merge3Block);

    B.CreateCall(barrierFunc);

    // cond 64

    BasicBlock* Cond4Block = createBasicBlock("cond4", reduceFunc);
    BasicBlock* Merge4Block = createBasicBlock("merge4", reduceFunc);

    cond = B.CreateICmpULT(threadId, i64);
    B.CreateCondBr(cond, Cond4Block, Merge4Block);
    B.SetInsertPoint(Cond4Block);

    val = B.CreateLoad(ideasAddr(sPtr));
    Value* idx64 = B.CreateGEP(reduceShared, B.CreateAdd(threadId, i64));
    Value* v64 = B.CreateLoad(ideasAddr(idx64));
    val = reduceOp(B, reduceType, val, v64);
    B.CreateStore(val, ideasAddr(sPtr));
    B.CreateStore(val, idxAddr);
    B.CreateBr(Merge4Block);

    B.SetInsertPoint(Merge4Block);

    B.CreateCall(barrierFunc);

    // cond 32

    BasicBlock* Cond5Block = createBasicBlock("cond5", reduceFunc);
    BasicBlock* Merge5Block = createBasicBlock("merge5", reduceFunc);

    cond = B.CreateICmpULT(threadId, i32);
    B.CreateCondBr(cond, Cond5Block, Merge5Block);
    B.SetInsertPoint(Cond5Block);

    val = B.CreateLoad(ideasAddr(sPtr));
    Value* idx32 = B.CreateGEP(reduceShared, B.CreateAdd(threadId, i32));
    Value* v32 = B.CreateLoad(ideasAddr(idx32));
    val = reduceOp(B, reduceType, val, v32);
    B.CreateStore(val, ideasAddr(sPtr));
    B.CreateStore(val, idxAddr);

    B.CreateCall(barrierFunc);

    // cond 16

    val = B.CreateLoad(ideasAddr(sPtr));
    Value* idx16 = B.CreateGEP(reduceShared, B.CreateAdd(threadId, i16));
    Value* v16 = B.CreateLoad(ideasAddr(idx16));
    val = reduceOp(B, reduceType, val, v16);
    B.CreateStore(val, ideasAddr(sPtr));
    B.CreateStore(val, idxAddr);

    B.CreateCall(barrierFunc);

    // cond 8

    val = B.CreateLoad(ideasAddr(sPtr));
    Value* idx8 = B.CreateGEP(reduceShared, B.CreateAdd(threadId, i8));
    Value* v8 = B.CreateLoad(ideasAddr(idx8));
    val = reduceOp(B, reduceType, val, v8);
    B.CreateStore(val, ideasAddr(sPtr));
    B.CreateStore(val, idxAddr);

    B.CreateCall(barrierFunc);

    // cond 4

    val = B.CreateLoad(ideasAddr(sPtr));
    Value* idx4 = B.CreateGEP(reduceShared, B.CreateAdd(threadId, i4));
    Value* v4 = B.CreateLoad(ideasAddr(idx4));
    val = reduceOp(B, reduceType, val, v4);
    B.CreateStore(val, ideasAddr(sPtr));
    B.CreateStore(val, idxAddr);

    B.CreateCall(barrierFunc);

    // cond 2

    val = B.CreateLoad(ideasAddr(sPtr));
    Value* idx2 = B.CreateGEP(reduceShared, B.CreateAdd(threadId, i2));
    Value* v2 = B.CreateLoad(ideasAddr(idx2));
    val = reduceOp(B, reduceType, val, v2);
    B.CreateStore(val, ideasAddr(sPtr));
    B.CreateStore(val, idxAddr);

    B.CreateCall(barrierFunc);

    // cond 1

    val = B.CreateLoad(ideasAddr(sPtr));
    Value* idx1 = B.CreateGEP(reduceShared, B.CreateAdd(threadId, i1));
    Value* v1 = B.CreateLoad(ideasAddr(idx1));
    val = reduceOp(B, reduceType, val, v1);
    B.CreateStore(val, ideasAddr(sPtr));
    B.CreateStore(val, idxAddr);

    B.CreateCall(barrierFunc);


    // -------------------

    BasicBlock* Cond12Block = createBasicBlock("cond12", reduceFunc);
    BasicBlock* Merge12Block = createBasicBlock("merge12", reduceFunc);

    cond = B.CreateICmpEQ(threadId, i0);
    B.CreateCondBr(cond, Cond12Block, Merge12Block);
    B.SetInsertPoint(Cond12Block);

    Value* idx0 = B.CreateGEP(reduceShared, i0);
    val = B.CreateLoad(ideasAddr(sPtr));
    Value* idxOut = B.CreateGEP(reduceArray, blockIdx);
    B.CreateStore(val, ideasAddr(idxOut));
    B.CreateBr(Merge12Block);

    B.SetInsertPoint(Merge12Block);

    B.CreateBr(Merge5Block);

    B.SetInsertPoint(Merge5Block);

    B.CreateRetVoid();
  }

  //reduceFunc->dump();

  // ===========================================

  CurFn = prevFn;
  B.SetInsertPoint(startBlock, startPoint);

  AllocaInsertPt = prevAllocaInsertPt;

  NamedMDNode* annotations = 
    ptxModule.getOrInsertNamedMetadata("nvvm.annotations");
  
  SmallVector<Metadata*, 3> av;

  if(reduceFunc){
    av.push_back(ValueAsMetadata::get(reduceFunc));
  }
  else{
    av.push_back(ValueAsMetadata::get(func));    
  }

  av.push_back(MDString::get(ptxModule.getContext(), "kernel"));
  av.push_back(ValueAsMetadata::get(llvm::ConstantInt::get(KR.Int32Ty, 1)));

  annotations->addOperand(MDNode::get(ptxModule.getContext(), av));

  //llvm::errs() << "---------------------- ptx module\n";
  //ptxModule.dump();

  const llvm::Target* target = nullptr;

  for(TargetRegistry::iterator itr =  TargetRegistry::targets().begin(),
      itrEnd =  TargetRegistry::targets().end(); itr != itrEnd; ++itr){
    if(string(itr->getName()) == "nvptx64"){
      target = &*itr;
    }
  }

  assert(target && "failed to find NVPTX target");

  Triple triple(sys::getDefaultTargetTriple());
  triple.setArch(Triple::nvptx64);
    
  TargetMachine* targetMachine =  
      target->createTargetMachine(triple.getTriple(),
                                  //"sm_35",
                                  "sm_35",
                                  "",
                                  llvm::TargetOptions(),
                                  Reloc::Default,
                                  CodeModel::Default,
                                  CodeGenOpt::Aggressive);

  DataLayout layout("e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:"
    "64:64-f32:32:32-f64:64:64-v16:16:16-v32:32:32-v64:"
    "64:64-v128:128:128-n16:32:64");

  ptxModule.setDataLayout(layout);

  llvm::legacy::PassManager* passManager = new legacy::PassManager;

  passManager->add(createVerifierPass());

  StringMap<int> ReflectParams;
  ReflectParams["__CUDA_FTZ"] = 1;
  passManager->add(createNVVMReflectPass(ReflectParams));

  passManager->add(createInstructionCombiningPass());
  passManager->add(createReassociatePass());
  passManager->add(createGVNPass());
  passManager->add(createCFGSimplificationPass());
  //passManager->add(createSLPVectorizePass());
  passManager->add(createBreakCriticalEdgesPass());
  passManager->add(createConstantPropagationPass());
  passManager->add(createDeadInstEliminationPass());
  passManager->add(createDeadStoreEliminationPass());
  passManager->add(createInstructionCombiningPass());
  passManager->add(createCFGSimplificationPass());

  SmallVector<char, 65536> buf;
  raw_svector_ostream ostr(buf);
  
  bool fail =
  targetMachine->addPassesToEmitFile(*passManager,
                                     ostr,
                                     TargetMachine::CGFT_AssemblyFile,
                                     false);

  assert(!fail);
  
  passManager->run(ptxModule);
      
  delete passManager;

  string ptx = ostr.str().str();

  //ndump(ptx);

  Constant* pc = ConstantDataArray::getString(C, ptx);
  
  GlobalVariable* ptxGlobal = 
    new GlobalVariable(CGM.getModule(),
                       pc->getType(),
                       true,
                       GlobalValue::PrivateLinkage,
                       pc,
                       "ptx");

  Value* kernelId = ConstantInt::get(Int32Ty, nextKernelId_++);

  Value* ptxStr = B.CreateBitCast(ptxGlobal, VoidPtrTy);

  ValueVec args = {kernelId, ptxStr};

  llvm::Type* i1Ty = llvm::IntegerType::getInt1Ty(C);

  Constant* falseVal = ConstantInt::get(i1Ty, 0);
  Constant* trueVal = ConstantInt::get(i1Ty, 1);

  if(reduceVar){
    QualType t = reduceVar->getType().getNonReferenceType();

    size_t bytes = ConvertType(t)->getPrimitiveSizeInBits()/8;

    args.push_back(ConstantInt::get(Int32Ty, bytes));

    if(ConvertType(t)->isFloatingPointTy()){
      args.push_back(trueVal);
      args.push_back(falseVal);
    }
    else{
      args.push_back(falseVal);
      if(t.getTypePtr()->isSignedIntegerType()){
        args.push_back(trueVal);
      }
    }

    if(reduceType == ReduceType::Sum){
      args.push_back(trueVal);
    }
    else{
      args.push_back(falseVal);
    }
  }
  else{
    args.push_back(ConstantInt::get(Int32Ty, 0));
    args.push_back(falseVal);
    args.push_back(falseVal);
    args.push_back(falseVal);
  }

  Value* ready = B.CreateCall(R.CudaInitKernelFunc(), args);

  BasicBlock* initBlock = createBasicBlock("parallel_for.init", startFunc);
  BasicBlock* readyBlock = createBasicBlock("parallel_for.ready", startFunc);

  B.CreateCondBr(ready, readyBlock, initBlock);

  B.SetInsertPoint(initBlock);

  for(const VarDecl* vd : viewVars){
    ViewInfo& info = viewInfoMap_[vd];

    llvm::Type* t = ConvertType(info.elementType);
    uint32_t elementSize = t->getPrimitiveSizeInBits()/8;

    Address addr = GetAddrOfLocalVar(vd);
    assert(addr.isValid());
    
    Value* viewPtr = B.CreateBitCast(addr.getPointer(), VoidPtrTy);
    
    uint32_t flags = 0;
    
    if(readViewVars.find(vd) != readViewVars.end()){
      flags |= 0x01;
    }
    
    if(writeViewVars.find(vd) != writeViewVars.end()){
      flags |= 0x02;
    }

    Value* staticDims = ConstantInt::get(Int32Ty, info.staticSizes.size());
    Value* staticSizes = B.CreateAlloca(Int32Ty, staticDims);

    size_t i = 0;
    for(uint32_t ss : info.staticSizes){
      Address sizePtr = ideasAddr(B.CreateConstGEP1_32(staticSizes, i));
      B.CreateStore(ConstantInt::get(Int32Ty, ss), sizePtr);
      ++i;
    }

    ValueVec args = 
      {kernelId, viewPtr, ConstantInt::get(Int32Ty, elementSize), 
       staticDims,
       staticSizes,
       ConstantInt::get(Int32Ty, info.runtimeDims),
       ConstantInt::get(Int32Ty, flags)};

    B.CreateCall(R.CudaAddViewFunc(), args);
  }

  for(const VarDecl* vd : arrayVars){
    ArrayInfo& info = arrayInfoMap_[vd];

    llvm::Type* t = ConvertType(info.elementType);
    uint32_t elementSize = t->getPrimitiveSizeInBits()/8;

    Address addr = GetAddrOfLocalVar(vd);
    assert(addr.isValid());

    Value* ptr = B.CreateLoad(addr);

    Value* arrayPtr = B.CreateBitCast(ptr, VoidPtrTy);
    
    uint32_t flags = 0;
    
    if(readArrayVars.find(vd) != readArrayVars.end()){
      flags |= 0x01;
    }
    
    if(writeArrayVars.find(vd) != writeArrayVars.end()){
      flags |= 0x02;
    }

    ValueVec args = 
      {kernelId, arrayPtr, ConstantInt::get(Int32Ty, elementSize), 
       info.size, ConstantInt::get(Int32Ty, flags)};

    B.CreateCall(R.CudaAddArrayFunc(), args);
  }

  for(const VarDecl* vd : captureVars){
    auto itr = LocalDeclMap.find(vd);
    assert(itr != LocalDeclMap.end());

    Value* varPtr = B.CreateBitCast(itr->second.getPointer(), VoidPtrTy);

    B.CreateCall(R.CudaAddVarFunc(), {kernelId, varPtr});  
  }

  for(const VarDecl* vd : arrayVars){
    parallelForParamMap_.erase(vd);
  }

  B.CreateBr(readyBlock);

  B.SetInsertPoint(readyBlock);

  args = {kernelId, nv};

  if(reduceVar){
    auto dr = dyn_cast<DeclRefExpr>(E->getArg(2));
    assert(dr && "failed to get reduce result decl");

    auto itr = LocalDeclMap.find(dr->getDecl());
    assert(itr != LocalDeclMap.end());

    Value* reduceRetPtr = B.CreateBitCast(itr->second.getPointer(), VoidPtrTy);
    args.push_back(reduceRetPtr);
  }
  else{
    args.push_back(ConstantPointerNull::get(VoidPtrTy));
  }

  B.CreateCall(R.CudaRunKernelFunc(), args);

  parallelForStack_.pop_back();

  //llvm::errs() << "---------------------- host module\n";
  //CGM.getModule().dump();

  //ndump(ptx); 
}

void CodeGenFunction::EmitParallelConstructPTX2(const CallExpr* E){
  assert(false);

  /*
  using namespace llvm;
  using namespace std;

  using ValueVec = vector<Value*>;
  using TypeVec = vector<llvm::Type*>;
  
  auto& B = Builder;
  auto& R = CGM.getIdeasRuntime();
  LLVMContext& C = getLLVMContext();

  const VarDecl* reduceVar;

  const LambdaExpr* le = GetLambda(E->getArg(1));

  assert(le && "expected a lambda");
  
  vector<const VarDecl*> captureVars;

  for(auto ci : le->captures()){
    bool shouldCapture = true;

    const VarDecl* vd = ci.getCapturedVar();

    QualType ct = vd->getType().getCanonicalType();
    if(const RecordType* rt = dyn_cast<RecordType>(ct.getTypePtr())){
      (void)rt;
      if(ct.getAsString().find("class Kokkos::View") == 0){
        shouldCapture = false;
      }
    }

    if(shouldCapture){
      captureVars.push_back(vd);
    }
  } 

  CXXMethodDecl* md = le->getCallOperator();

  const FunctionDecl* f = E->getDirectCallee();
  assert(f);

  if(f->getQualifiedNameAsString() == "Kokkos::parallel_for"){
    reduceVar = nullptr;
    assert(md->getNumParams() == 1);
    assert(E->getNumArgs() == 3);
  }
  else if(f->getQualifiedNameAsString() == "Kokkos::parallel_reduce"){
    reduceVar = md->getParamDecl(1);
    assert(md->getNumParams() == 2);
    assert(E->getNumArgs() == 5); 
  }
  else{
    assert(false && "expected parallel for or reduce");
  }

  ParallelForInfo* parent;
  ParallelForInfo* info;

  PTXParallelConstructVisitor::VarSet viewVars;
  PTXParallelConstructVisitor::VarSet readViewVars;
  PTXParallelConstructVisitor::VarSet writeViewVars;

  ReduceType reduceType = ReduceType::None;

  if(parallelForStack_.empty()){
    ParallelForVisitor visitor;
    visitor.VisitStmt(const_cast<CallExpr*>(E));
    parent = nullptr;
    info = visitor.getParallelForInfo();

    PTXParallelConstructVisitor ptxVisitor(reduceVar);
    ptxVisitor.VisitStmt(const_cast<CallExpr*>(E));
    viewVars = ptxVisitor.viewVars();
    readViewVars = ptxVisitor.readViewVars();
    writeViewVars = ptxVisitor.writeViewVars();

    if(reduceVar){      
      for(auto op : ptxVisitor.reduceOps()){
        if(auto bo = dyn_cast<BinaryOperator>(op)){
          if(bo->getOpcode() == BO_AddAssign){
            
            assert(reduceType == ReduceType::None || 
                   reduceType == ReduceType::Sum);
            
            reduceType = ReduceType::Sum;
          }
          else if(bo->getOpcode() == BO_MulAssign){

            assert(reduceType == ReduceType::None || 
                   reduceType == ReduceType::Product);

            reduceType = ReduceType::Product;
          }
          else{
            assert(false && "invalid reduce type");
          }
        }
        else if(auto uo = dyn_cast<UnaryOperator>(op)){
          (void)uo;
          assert(reduceType == ReduceType::None || 
                 reduceType == ReduceType::Sum);
          
          reduceType = ReduceType::Sum;
        }
      }

      assert(reduceType != ReduceType::None && "failed to find reduce operator");
    }
  }
  else{
    parent = parallelForStack_.back();

    for(auto i : parent->children){
      if(i->callExpr == E){
        info = i;
        break;
      }
    }
    assert(info);
  }

  for(const VarDecl* vd : viewVars){    
    ViewInfo info;

    const RecordType* rt = 
      dyn_cast<RecordType>(vd->getType().getCanonicalType().getTypePtr());
    assert(rt && "expected a RecordType");

    const ClassTemplateSpecializationDecl* td = 
      dyn_cast<ClassTemplateSpecializationDecl>(rt->getDecl());
    assert(td && "expected a ClassTemplateSpecializationDecl");

    const TemplateArgumentList& tl = td->getTemplateArgs();
    TemplateArgument ta = tl[0];
    QualType tt = ta.getAsType();

    const PointerType* pt;
    const ConstantArrayType* at;

    for(;;){
      if((at = dyn_cast<ConstantArrayType>(tt.getTypePtr()))){
        info.staticSizes.push_back(at->getSize().getZExtValue());
        tt = at->getElementType();
      }
      else if((pt = dyn_cast<PointerType>(tt.getTypePtr()))){
        break;
      }
      else{
        assert(false && "expected a ConstantArrayType or PointerType");        
      }
    }

    info.runtimeDims = 1;

    for(;;){
      QualType et = pt->getPointeeType();

      pt = dyn_cast<PointerType>(et.getTypePtr());

      if(!pt){
        const BuiltinType* bt = dyn_cast<BuiltinType>(et.getTypePtr());
        assert(bt && "expected a BultinType");

        info.elementType = et;
        break;
      }

      ++info.runtimeDims;
    }

    viewInfoMap_[vd] = info;
  }

  parallelForStack_.push_back(info);

  BasicBlock* startBlock = B.GetInsertBlock(); 
  BasicBlock::iterator startPoint = B.GetInsertPoint();
  Function* startFunc = startBlock->getParent(); 
    
  Value* nv = EmitScalarExpr(E->getArg(0));

  nv = B.CreateTrunc(nv, Int32Ty);

  CompoundStmt* body = le->getBody();
  
  ParmVarDecl* indexVar = md->getParamDecl(0);

  //llvm::Type* indexType = ConvertType(indexVar->getType());

  TypeVec params;

  for(const VarDecl* vd : viewVars){
    llvm::Type* t = ConvertType(viewInfoMap_[vd].elementType);
    params.push_back(llvm::PointerType::get(t, 0));
    params.push_back(llvm::PointerType::get(Int32Ty, 0));
  }

  for(const VarDecl* vd : captureVars){
    llvm::Type* t = ConvertType(vd->getType());
    params.push_back(t);
  }

  llvm::FunctionType* ft;

  if(reduceVar){
    llvm::Type* rt = ConvertType(reduceVar->getType().getNonReferenceType());
    params.push_back(Int32Ty);
    ft = llvm::FunctionType::get(rt, params, false);
    // shared memory
    //params.push_back(llvm::PointerType::get(rt, 0));
  }
  else{
    params.push_back(Int32Ty);
    ft = llvm::FunctionType::get(VoidTy, params, false);
  }

  llvm::Module ptxModule("PTXModule", C);
  CGIdeasRuntime KR(ptxModule);

  llvm::Function* func =
    llvm::Function::Create(ft,
                           llvm::Function::ExternalLinkage,
                           "run",
                           &ptxModule);

  //func->dump();

  auto aitr = func->arg_begin();

  parallelForParamMap_.clear();
  
  Value* reduceArray;

  for(const VarDecl* vd : viewVars){ 
    aitr->setName(vd->getName());
    parallelForParamMap_[vd] = aitr;
    ++aitr;

    string pn = vd->getName();
    pn += ".dims";
      
    aitr->setName(pn);

    parallelForParamDimMap_[vd] = aitr;
    ++aitr;
  }

  auto aitrCapture = aitr;

  for(const VarDecl* vd : captureVars){
    aitr->setName(vd->getName());
    ++aitr;
  }

  Value* count;
  if(reduceVar){
    aitr->setName("index");
  }
  else{
    count = aitr;
    aitr->setName("n");
    ++aitr;
  }

  llvm::Function* prevFn = CurFn;
  CurFn = func;

  BasicBlock* entry = createBasicBlock("entry", func);

  auto prevAllocaInsertPt = AllocaInsertPt;

  llvm::Value* Undef = llvm::UndefValue::get(Int32Ty);
  AllocaInsertPt = new llvm::BitCastInst(Undef, Int32Ty, "", entry);

  ReturnBlock = getJumpDestInCurrentScope("return");

  B.SetInsertPoint(entry);

  for(const VarDecl* vd : captureVars){
    Value* varPtr = B.CreateAlloca(ConvertType(vd->getType()));
    B.CreateStore(aitrCapture++, ideasAddr(varPtr));
    parallelForParamMap_[vd] = varPtr;
  }

  Value* reducePtr2;

  if(reduceVar){
    llvm::Type* pt = ConvertType(reduceVar->getType());
    llvm::Type* t = ConvertType(reduceVar->getType().getNonReferenceType());

    reducePtr2 = B.CreateAlloca(pt);
    Value* ptr5 = B.CreateAlloca(t);
    B.CreateStore(ptr5, ideasAddr(reducePtr2));
    Value* indexPtr = B.CreateAlloca(Int32Ty);

    Value* initVal;
    
    if(t->isFloatingPointTy()){
      if(reduceType == ReduceType::Sum){
        initVal = ConstantFP::get(t, 0.0);
      }
      else if(reduceType == ReduceType::Product){
        initVal = ConstantFP::get(t, 1.0);
      }
      else{
        assert(false && "invalid reduce type");
      }
    }
    else{
      if(reduceType == ReduceType::Sum){
        initVal = ConstantInt::get(t, 0);
      }
      else if(reduceType == ReduceType::Product){
        initVal = ConstantInt::get(t, 1);
      }
      else{
        assert(false && "invalid reduce type");
      }
    }

    Value* reducePtr = B.CreateLoad(ideasAddr(reducePtr2));
    B.CreateStore(initVal, ideasAddr(reducePtr));
    setAddrOfLocalVar(reduceVar, ideasAddr(reducePtr2));

    setAddrOfLocalVar(indexVar, ideasAddr(indexPtr));  
  }
  else{
    Value* threadIdx = B.CreateCall(KR.getSREGFunc("tid.x"));
    Value* blockIdx = B.CreateCall(KR.getSREGFunc("ctaid.x"));
    Value* blockDim = B.CreateCall(KR.getSREGFunc("ntid.x"));

    Value* threadId = 
      B.CreateAdd(threadIdx, B.CreateMul(blockIdx, blockDim), "threadId");

    Value* cond = 
      B.CreateICmpUGT(threadId, count, "cond");

    BasicBlock* RetBlock = createBasicBlock("parallel_for.ret", func);

    BasicBlock* ContBlock = createBasicBlock("parallel_for.cont", func);

    B.CreateCondBr(cond, RetBlock, ContBlock);
    
    B.SetInsertPoint(RetBlock);

    B.CreateRetVoid();

    B.SetInsertPoint(ContBlock);

    Address threadIdPtr = 
    ideasAddr(B.CreateAlloca(threadId->getType(), nullptr, "threadId.ptr"));

    B.CreateStore(threadId, threadIdPtr);

    setAddrOfLocalVar(indexVar, threadIdPtr);
  }
    
  EmitStmt(body);

  if(reduceVar){
    Value* retPtr = B.CreateLoad(ideasAddr(reducePtr2));
    Value* ret = B.CreateLoad(ideasAddr(retPtr));
    B.CreateRet(ret);
  }
  else{
    B.CreateRetVoid();
  }

  //llvm::errs() << "---------------- kernel func\n";
  //func->dump();

  //reduceFunc->dump();

  // ===========================================

  CurFn = prevFn;
  B.SetInsertPoint(startBlock, startPoint);

  AllocaInsertPt = prevAllocaInsertPt;

  //llvm::errs() << "---------------------- ptx module\n";
  //ptxModule.dump();

  const llvm::Target* target = nullptr;

  for(TargetRegistry::iterator itr =  TargetRegistry::targets().begin(),
      itrEnd =  TargetRegistry::targets().end(); itr != itrEnd; ++itr){
    if(string(itr->getName()) == "nvptx64"){
      target = &*itr;
    }
  }

  assert(target && "failed to find NVPTX target");

  Triple triple(sys::getDefaultTargetTriple());
  triple.setArch(Triple::nvptx64);
    
  TargetMachine* targetMachine =  
      target->createTargetMachine(triple.getTriple(),
                                  "sm_35",
                                  "",
                                  llvm::TargetOptions(),
                                  Reloc::Default,
                                  CodeModel::Default,
                                  CodeGenOpt::Aggressive);

  DataLayout layout("e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:"
    "64:64-f32:32:32-f64:64:64-v16:16:16-v32:32:32-v64:"
    "64:64-v128:128:128-n16:32:64");

  ptxModule.setDataLayout(layout);

  llvm::legacy::PassManager* passManager = new legacy::PassManager;

  passManager->add(createVerifierPass());

  StringMap<int> ReflectParams;
  ReflectParams["__CUDA_FTZ"] = 1;
  passManager->add(createNVVMReflectPass(ReflectParams));

  passManager->add(createInstructionCombiningPass());
  passManager->add(createReassociatePass());
  passManager->add(createGVNPass());
  passManager->add(createCFGSimplificationPass());
  //passManager->add(createSLPVectorizePass());
  passManager->add(createBreakCriticalEdgesPass());
  passManager->add(createConstantPropagationPass());
  passManager->add(createDeadInstEliminationPass());
  passManager->add(createDeadStoreEliminationPass());
  passManager->add(createInstructionCombiningPass());
  passManager->add(createCFGSimplificationPass());

  SmallVector<char, 65536> buf;
  raw_svector_ostream ostr(buf);
  
  bool fail =
  targetMachine->addPassesToEmitFile(*passManager,
                                     ostr,
                                     TargetMachine::CGFT_AssemblyFile,
                                     false);

  assert(!fail);
  
  passManager->run(ptxModule);
      
  delete passManager;

  string ptx = ostr.str().str();

  //ndump(ptx);

  Constant* pc = ConstantDataArray::getString(C, ptx);
  
  GlobalVariable* ptxGlobal = 
    new GlobalVariable(CGM.getModule(),
                       pc->getType(),
                       true,
                       GlobalValue::PrivateLinkage,
                       pc,
                       "ptx");

  Value* kernelId = ConstantInt::get(Int32Ty, nextKernelId_++);

  Value* ptxStr = B.CreateBitCast(ptxGlobal, VoidPtrTy);

  ValueVec args = {kernelId, ptxStr};

  llvm::Type* i1Ty = llvm::IntegerType::getInt1Ty(C);

  Constant* falseVal = ConstantInt::get(i1Ty, 0);
  Constant* trueVal = ConstantInt::get(i1Ty, 1);

  if(reduceVar){
    QualType t = reduceVar->getType().getNonReferenceType();

    size_t bytes = ConvertType(t)->getPrimitiveSizeInBits()/8;

    args.push_back(ConstantInt::get(Int32Ty, bytes));

    if(ConvertType(t)->isFloatingPointTy()){
      args.push_back(trueVal);
      args.push_back(falseVal);
    }
    else{
      args.push_back(falseVal);
      if(t.getTypePtr()->isSignedIntegerType()){
        args.push_back(trueVal);
      }
    }

    if(reduceType == ReduceType::Sum){
      args.push_back(trueVal);
    }
    else{
      args.push_back(falseVal);
    }
  }
  else{
    args.push_back(ConstantInt::get(Int32Ty, 0));
    args.push_back(falseVal);
    args.push_back(falseVal);
    args.push_back(falseVal);
  }

  Value* ready = B.CreateCall(R.CudaInitKernel2Func(), args);

  BasicBlock* initBlock = createBasicBlock("parallel_for.init", startFunc);
  BasicBlock* readyBlock = createBasicBlock("parallel_for.ready", startFunc);

  B.CreateCondBr(ready, readyBlock, initBlock);

  B.SetInsertPoint(initBlock);

  for(const VarDecl* vd : viewVars){
    ViewInfo& info = viewInfoMap_[vd];

    llvm::Type* t = ConvertType(info.elementType);
    uint32_t elementSize = t->getPrimitiveSizeInBits()/8;

    Address addr = GetAddrOfLocalVar(vd);
    assert(addr.isValid());
    
    Value* viewPtr = B.CreateBitCast(addr.getPointer(), VoidPtrTy);
    
    uint32_t flags = 0;
    
    if(readViewVars.find(vd) != readViewVars.end()){
      flags |= 0x01;
    }
    
    if(writeViewVars.find(vd) != writeViewVars.end()){
      flags |= 0x02;
    }

    Value* staticDims = ConstantInt::get(Int32Ty, info.staticSizes.size());
    Value* staticSizes = B.CreateAlloca(Int32Ty, staticDims);

    size_t i = 0;
    for(uint32_t ss : info.staticSizes){
      Address sizePtr = ideasAddr(B.CreateConstGEP1_32(staticSizes, i));
      B.CreateStore(ConstantInt::get(Int32Ty, ss), sizePtr);
      ++i;
    }

    ValueVec args = 
      {kernelId, viewPtr, ConstantInt::get(Int32Ty, elementSize), 
       staticDims,
       staticSizes,
       ConstantInt::get(Int32Ty, info.runtimeDims),
       ConstantInt::get(Int32Ty, flags)};

    B.CreateCall(R.CudaAddViewFunc(), args);
  }

  for(const VarDecl* vd : captureVars){
    auto itr = LocalDeclMap.find(vd);
    assert(itr != LocalDeclMap.end());

    Value* varPtr = B.CreateBitCast(itr->second.getPointer(), VoidPtrTy);

    B.CreateCall(R.CudaAddVarFunc(), {kernelId, varPtr});  
  }

  B.CreateBr(readyBlock);

  B.SetInsertPoint(readyBlock);

  args = {kernelId, nv};

  if(reduceVar){
    auto dr = dyn_cast<DeclRefExpr>(E->getArg(2));
    assert(dr && "failed to get reduce result decl");

    auto itr = LocalDeclMap.find(dr->getDecl());
    assert(itr != LocalDeclMap.end());

    Value* reduceRetPtr = B.CreateBitCast(itr->second.getPointer(), VoidPtrTy);
    args.push_back(reduceRetPtr);
  }
  else{
    args.push_back(ConstantPointerNull::get(VoidPtrTy));
  }

  B.CreateCall(R.CudaRunKernel2Func(), args);

  parallelForStack_.pop_back();

  //llvm::errs() << "---------------------- host module\n";
  //CGM.getModule().dump();

  //ndump(ptx);
  */ 
}

// ===========================================

//===----------------------------------------------------------------------===//
//                              Statement Emission
//===----------------------------------------------------------------------===//

void CodeGenFunction::EmitStopPoint(const Stmt *S) {
  if (CGDebugInfo *DI = getDebugInfo()) {
    SourceLocation Loc;
    Loc = S->getLocStart();
    DI->EmitLocation(Builder, Loc);

    LastStopPoint = Loc;
  }
}

void CodeGenFunction::EmitStmt(const Stmt *S) {
  assert(S && "Null statement?");
  PGO.setCurrentStmt(S);

  // These statements have their own debug info handling.
  if (EmitSimpleStmt(S))
    return;

  // Check if we are generating unreachable code.
  if (!HaveInsertPoint()) {
    // If so, and the statement doesn't contain a label, then we do not need to
    // generate actual code. This is safe because (1) the current point is
    // unreachable, so we don't need to execute the code, and (2) we've already
    // handled the statements which update internal data structures (like the
    // local variable map) which could be used by subsequent statements.
    if (!ContainsLabel(S)) {
      // Verify that any decl statements were handled as simple, they may be in
      // scope of subsequent reachable statements.
      assert(!isa<DeclStmt>(*S) && "Unexpected DeclStmt!");
      return;
    }

    // Otherwise, make a new block to hold the code.
    EnsureInsertPoint();
  }

  // Generate a stoppoint if we are emitting debug info.
  EmitStopPoint(S);

  switch (S->getStmtClass()) {
  case Stmt::NoStmtClass:
  case Stmt::CXXCatchStmtClass:
  case Stmt::SEHExceptStmtClass:
  case Stmt::SEHFinallyStmtClass:
  case Stmt::MSDependentExistsStmtClass:
    llvm_unreachable("invalid statement class to emit generically");
  case Stmt::NullStmtClass:
  case Stmt::CompoundStmtClass:
  case Stmt::DeclStmtClass:
  case Stmt::LabelStmtClass:
  case Stmt::AttributedStmtClass:
  case Stmt::GotoStmtClass:
  case Stmt::BreakStmtClass:
  case Stmt::ContinueStmtClass:
  case Stmt::DefaultStmtClass:
  case Stmt::CaseStmtClass:
  case Stmt::SEHLeaveStmtClass:
    llvm_unreachable("should have emitted these statements as simple");

#define STMT(Type, Base)
#define ABSTRACT_STMT(Op)
#define EXPR(Type, Base) \
  case Stmt::Type##Class:
#include "clang/AST/StmtNodes.inc"
  {
    // Remember the block we came in on.
    llvm::BasicBlock *incoming = Builder.GetInsertBlock();
    assert(incoming && "expression emission must have an insertion point");

    EmitIgnoredExpr(cast<Expr>(S));

    llvm::BasicBlock *outgoing = Builder.GetInsertBlock();
    assert(outgoing && "expression emission cleared block!");

    // The expression emitters assume (reasonably!) that the insertion
    // point is always set.  To maintain that, the call-emission code
    // for noreturn functions has to enter a new block with no
    // predecessors.  We want to kill that block and mark the current
    // insertion point unreachable in the common case of a call like
    // "exit();".  Since expression emission doesn't otherwise create
    // blocks with no predecessors, we can just test for that.
    // However, we must be careful not to do this to our incoming
    // block, because *statement* emission does sometimes create
    // reachable blocks which will have no predecessors until later in
    // the function.  This occurs with, e.g., labels that are not
    // reachable by fallthrough.
    if (incoming != outgoing && outgoing->use_empty()) {
      outgoing->eraseFromParent();
      Builder.ClearInsertionPoint();
    }
    break;
  }

  case Stmt::IndirectGotoStmtClass:
    EmitIndirectGotoStmt(cast<IndirectGotoStmt>(*S)); break;

  case Stmt::IfStmtClass:       EmitIfStmt(cast<IfStmt>(*S));             break;
  case Stmt::WhileStmtClass:    EmitWhileStmt(cast<WhileStmt>(*S));       break;
  case Stmt::DoStmtClass:       EmitDoStmt(cast<DoStmt>(*S));             break;
  case Stmt::ForStmtClass:      EmitForStmt(cast<ForStmt>(*S));           break;

  case Stmt::ReturnStmtClass:   EmitReturnStmt(cast<ReturnStmt>(*S));     break;

  case Stmt::SwitchStmtClass:   EmitSwitchStmt(cast<SwitchStmt>(*S));     break;
  case Stmt::GCCAsmStmtClass:   // Intentional fall-through.
  case Stmt::MSAsmStmtClass:    EmitAsmStmt(cast<AsmStmt>(*S));           break;
  case Stmt::CapturedStmtClass: {
    const CapturedStmt *CS = cast<CapturedStmt>(S);
    EmitCapturedStmt(*CS, CS->getCapturedRegionKind());
    }
    break;
  case Stmt::ObjCAtTryStmtClass:
    EmitObjCAtTryStmt(cast<ObjCAtTryStmt>(*S));
    break;
  case Stmt::ObjCAtCatchStmtClass:
    llvm_unreachable(
                    "@catch statements should be handled by EmitObjCAtTryStmt");
  case Stmt::ObjCAtFinallyStmtClass:
    llvm_unreachable(
                  "@finally statements should be handled by EmitObjCAtTryStmt");
  case Stmt::ObjCAtThrowStmtClass:
    EmitObjCAtThrowStmt(cast<ObjCAtThrowStmt>(*S));
    break;
  case Stmt::ObjCAtSynchronizedStmtClass:
    EmitObjCAtSynchronizedStmt(cast<ObjCAtSynchronizedStmt>(*S));
    break;
  case Stmt::ObjCForCollectionStmtClass:
    EmitObjCForCollectionStmt(cast<ObjCForCollectionStmt>(*S));
    break;
  case Stmt::ObjCAutoreleasePoolStmtClass:
    EmitObjCAutoreleasePoolStmt(cast<ObjCAutoreleasePoolStmt>(*S));
    break;

  case Stmt::CXXTryStmtClass:
    EmitCXXTryStmt(cast<CXXTryStmt>(*S));
    break;
  case Stmt::CXXForRangeStmtClass:
    EmitCXXForRangeStmt(cast<CXXForRangeStmt>(*S));
    break;
  case Stmt::SEHTryStmtClass:
    EmitSEHTryStmt(cast<SEHTryStmt>(*S));
    break;
  case Stmt::OMPParallelDirectiveClass:
    EmitOMPParallelDirective(cast<OMPParallelDirective>(*S));
    break;
  case Stmt::OMPSimdDirectiveClass:
    EmitOMPSimdDirective(cast<OMPSimdDirective>(*S));
    break;
  case Stmt::OMPForDirectiveClass:
    EmitOMPForDirective(cast<OMPForDirective>(*S));
    break;
  case Stmt::OMPForSimdDirectiveClass:
    EmitOMPForSimdDirective(cast<OMPForSimdDirective>(*S));
    break;
  case Stmt::OMPSectionsDirectiveClass:
    EmitOMPSectionsDirective(cast<OMPSectionsDirective>(*S));
    break;
  case Stmt::OMPSectionDirectiveClass:
    EmitOMPSectionDirective(cast<OMPSectionDirective>(*S));
    break;
  case Stmt::OMPSingleDirectiveClass:
    EmitOMPSingleDirective(cast<OMPSingleDirective>(*S));
    break;
  case Stmt::OMPMasterDirectiveClass:
    EmitOMPMasterDirective(cast<OMPMasterDirective>(*S));
    break;
  case Stmt::OMPCriticalDirectiveClass:
    EmitOMPCriticalDirective(cast<OMPCriticalDirective>(*S));
    break;
  case Stmt::OMPParallelForDirectiveClass:
    EmitOMPParallelForDirective(cast<OMPParallelForDirective>(*S));
    break;
  case Stmt::OMPParallelForSimdDirectiveClass:
    EmitOMPParallelForSimdDirective(cast<OMPParallelForSimdDirective>(*S));
    break;
  case Stmt::OMPParallelSectionsDirectiveClass:
    EmitOMPParallelSectionsDirective(cast<OMPParallelSectionsDirective>(*S));
    break;
  case Stmt::OMPTaskDirectiveClass:
    EmitOMPTaskDirective(cast<OMPTaskDirective>(*S));
    break;
  case Stmt::OMPTaskyieldDirectiveClass:
    EmitOMPTaskyieldDirective(cast<OMPTaskyieldDirective>(*S));
    break;
  case Stmt::OMPBarrierDirectiveClass:
    EmitOMPBarrierDirective(cast<OMPBarrierDirective>(*S));
    break;
  case Stmt::OMPTaskwaitDirectiveClass:
    EmitOMPTaskwaitDirective(cast<OMPTaskwaitDirective>(*S));
    break;
  case Stmt::OMPTaskgroupDirectiveClass:
    EmitOMPTaskgroupDirective(cast<OMPTaskgroupDirective>(*S));
    break;
  case Stmt::OMPFlushDirectiveClass:
    EmitOMPFlushDirective(cast<OMPFlushDirective>(*S));
    break;
  case Stmt::OMPOrderedDirectiveClass:
    EmitOMPOrderedDirective(cast<OMPOrderedDirective>(*S));
    break;
  case Stmt::OMPAtomicDirectiveClass:
    EmitOMPAtomicDirective(cast<OMPAtomicDirective>(*S));
    break;
  case Stmt::OMPTargetDirectiveClass:
    EmitOMPTargetDirective(cast<OMPTargetDirective>(*S));
    break;
  case Stmt::OMPTeamsDirectiveClass:
    EmitOMPTeamsDirective(cast<OMPTeamsDirective>(*S));
    break;
  case Stmt::OMPCancellationPointDirectiveClass:
    EmitOMPCancellationPointDirective(cast<OMPCancellationPointDirective>(*S));
    break;
  case Stmt::OMPCancelDirectiveClass:
    EmitOMPCancelDirective(cast<OMPCancelDirective>(*S));
    break;
  case Stmt::OMPTargetDataDirectiveClass:
    EmitOMPTargetDataDirective(cast<OMPTargetDataDirective>(*S));
    break;
  }
}

bool CodeGenFunction::EmitSimpleStmt(const Stmt *S) {
  switch (S->getStmtClass()) {
  default: return false;
  case Stmt::NullStmtClass: break;
  case Stmt::CompoundStmtClass: EmitCompoundStmt(cast<CompoundStmt>(*S)); break;
  case Stmt::DeclStmtClass:     EmitDeclStmt(cast<DeclStmt>(*S));         break;
  case Stmt::LabelStmtClass:    EmitLabelStmt(cast<LabelStmt>(*S));       break;
  case Stmt::AttributedStmtClass:
                            EmitAttributedStmt(cast<AttributedStmt>(*S)); break;
  case Stmt::GotoStmtClass:     EmitGotoStmt(cast<GotoStmt>(*S));         break;
  case Stmt::BreakStmtClass:    EmitBreakStmt(cast<BreakStmt>(*S));       break;
  case Stmt::ContinueStmtClass: EmitContinueStmt(cast<ContinueStmt>(*S)); break;
  case Stmt::DefaultStmtClass:  EmitDefaultStmt(cast<DefaultStmt>(*S));   break;
  case Stmt::CaseStmtClass:     EmitCaseStmt(cast<CaseStmt>(*S));         break;
  case Stmt::SEHLeaveStmtClass: EmitSEHLeaveStmt(cast<SEHLeaveStmt>(*S)); break;
  }

  return true;
}

/// EmitCompoundStmt - Emit a compound statement {..} node.  If GetLast is true,
/// this captures the expression result of the last sub-statement and returns it
/// (for use by the statement expression extension).
Address CodeGenFunction::EmitCompoundStmt(const CompoundStmt &S, bool GetLast,
                                          AggValueSlot AggSlot) {
  PrettyStackTraceLoc CrashInfo(getContext().getSourceManager(),S.getLBracLoc(),
                             "LLVM IR generation of compound statement ('{}')");

  // Keep track of the current cleanup stack depth, including debug scopes.
  LexicalScope Scope(*this, S.getSourceRange());

  return EmitCompoundStmtWithoutScope(S, GetLast, AggSlot);
}

Address
CodeGenFunction::EmitCompoundStmtWithoutScope(const CompoundStmt &S,
                                              bool GetLast,
                                              AggValueSlot AggSlot) {

  for (CompoundStmt::const_body_iterator I = S.body_begin(),
       E = S.body_end()-GetLast; I != E; ++I)
    EmitStmt(*I);

  Address RetAlloca = Address::invalid();
  if (GetLast) {
    // We have to special case labels here.  They are statements, but when put
    // at the end of a statement expression, they yield the value of their
    // subexpression.  Handle this by walking through all labels we encounter,
    // emitting them before we evaluate the subexpr.
    const Stmt *LastStmt = S.body_back();
    while (const LabelStmt *LS = dyn_cast<LabelStmt>(LastStmt)) {
      EmitLabel(LS->getDecl());
      LastStmt = LS->getSubStmt();
    }

    EnsureInsertPoint();

    QualType ExprTy = cast<Expr>(LastStmt)->getType();
    if (hasAggregateEvaluationKind(ExprTy)) {
      EmitAggExpr(cast<Expr>(LastStmt), AggSlot);
    } else {
      // We can't return an RValue here because there might be cleanups at
      // the end of the StmtExpr.  Because of that, we have to emit the result
      // here into a temporary alloca.
      RetAlloca = CreateMemTemp(ExprTy);
      EmitAnyExprToMem(cast<Expr>(LastStmt), RetAlloca, Qualifiers(),
                       /*IsInit*/false);
    }

  }

  return RetAlloca;
}

void CodeGenFunction::SimplifyForwardingBlocks(llvm::BasicBlock *BB) {
  llvm::BranchInst *BI = dyn_cast<llvm::BranchInst>(BB->getTerminator());

  // If there is a cleanup stack, then we it isn't worth trying to
  // simplify this block (we would need to remove it from the scope map
  // and cleanup entry).
  if (!EHStack.empty())
    return;

  // Can only simplify direct branches.
  if (!BI || !BI->isUnconditional())
    return;

  // Can only simplify empty blocks.
  if (BI != BB->begin())
    return;

  BB->replaceAllUsesWith(BI->getSuccessor(0));
  BI->eraseFromParent();
  BB->eraseFromParent();
}

void CodeGenFunction::EmitBlock(llvm::BasicBlock *BB, bool IsFinished) {
  llvm::BasicBlock *CurBB = Builder.GetInsertBlock();

  // Fall out of the current block (if necessary).
  EmitBranch(BB);

  if (IsFinished && BB->use_empty()) {
    delete BB;
    return;
  }

  // Place the block after the current block, if possible, or else at
  // the end of the function.
  if (CurBB && CurBB->getParent())
    CurFn->getBasicBlockList().insertAfter(CurBB, BB);
  else
    CurFn->getBasicBlockList().push_back(BB);
  Builder.SetInsertPoint(BB);
}

void CodeGenFunction::EmitBranch(llvm::BasicBlock *Target) {
  // Emit a branch from the current block to the target one if this
  // was a real block.  If this was just a fall-through block after a
  // terminator, don't emit it.
  llvm::BasicBlock *CurBB = Builder.GetInsertBlock();

  if (!CurBB || CurBB->getTerminator()) {
    // If there is no insert point or the previous block is already
    // terminated, don't touch it.
  } else {
    // Otherwise, create a fall-through branch.
    Builder.CreateBr(Target);
  }

  Builder.ClearInsertionPoint();
}

void CodeGenFunction::EmitBlockAfterUses(llvm::BasicBlock *block) {
  bool inserted = false;
  for (llvm::User *u : block->users()) {
    if (llvm::Instruction *insn = dyn_cast<llvm::Instruction>(u)) {
      CurFn->getBasicBlockList().insertAfter(insn->getParent(), block);
      inserted = true;
      break;
    }
  }

  if (!inserted)
    CurFn->getBasicBlockList().push_back(block);

  Builder.SetInsertPoint(block);
}

CodeGenFunction::JumpDest
CodeGenFunction::getJumpDestForLabel(const LabelDecl *D) {
  JumpDest &Dest = LabelMap[D];
  if (Dest.isValid()) return Dest;

  // Create, but don't insert, the new block.
  Dest = JumpDest(createBasicBlock(D->getName()),
                  EHScopeStack::stable_iterator::invalid(),
                  NextCleanupDestIndex++);
  return Dest;
}

void CodeGenFunction::EmitLabel(const LabelDecl *D) {
  // Add this label to the current lexical scope if we're within any
  // normal cleanups.  Jumps "in" to this label --- when permitted by
  // the language --- may need to be routed around such cleanups.
  if (EHStack.hasNormalCleanups() && CurLexicalScope)
    CurLexicalScope->addLabel(D);

  JumpDest &Dest = LabelMap[D];

  // If we didn't need a forward reference to this label, just go
  // ahead and create a destination at the current scope.
  if (!Dest.isValid()) {
    Dest = getJumpDestInCurrentScope(D->getName());

  // Otherwise, we need to give this label a target depth and remove
  // it from the branch-fixups list.
  } else {
    assert(!Dest.getScopeDepth().isValid() && "already emitted label!");
    Dest.setScopeDepth(EHStack.stable_begin());
    ResolveBranchFixups(Dest.getBlock());
  }

  EmitBlock(Dest.getBlock());
  incrementProfileCounter(D->getStmt());
}

/// Change the cleanup scope of the labels in this lexical scope to
/// match the scope of the enclosing context.
void CodeGenFunction::LexicalScope::rescopeLabels() {
  assert(!Labels.empty());
  EHScopeStack::stable_iterator innermostScope
    = CGF.EHStack.getInnermostNormalCleanup();

  // Change the scope depth of all the labels.
  for (SmallVectorImpl<const LabelDecl*>::const_iterator
         i = Labels.begin(), e = Labels.end(); i != e; ++i) {
    assert(CGF.LabelMap.count(*i));
    JumpDest &dest = CGF.LabelMap.find(*i)->second;
    assert(dest.getScopeDepth().isValid());
    assert(innermostScope.encloses(dest.getScopeDepth()));
    dest.setScopeDepth(innermostScope);
  }

  // Reparent the labels if the new scope also has cleanups.
  if (innermostScope != EHScopeStack::stable_end() && ParentScope) {
    ParentScope->Labels.append(Labels.begin(), Labels.end());
  }
}


void CodeGenFunction::EmitLabelStmt(const LabelStmt &S) {
  EmitLabel(S.getDecl());
  EmitStmt(S.getSubStmt());
}

void CodeGenFunction::EmitAttributedStmt(const AttributedStmt &S) {
  const Stmt *SubStmt = S.getSubStmt();
  switch (SubStmt->getStmtClass()) {
  case Stmt::DoStmtClass:
    EmitDoStmt(cast<DoStmt>(*SubStmt), S.getAttrs());
    break;
  case Stmt::ForStmtClass:
    EmitForStmt(cast<ForStmt>(*SubStmt), S.getAttrs());
    break;
  case Stmt::WhileStmtClass:
    EmitWhileStmt(cast<WhileStmt>(*SubStmt), S.getAttrs());
    break;
  case Stmt::CXXForRangeStmtClass:
    EmitCXXForRangeStmt(cast<CXXForRangeStmt>(*SubStmt), S.getAttrs());
    break;
  default:
    EmitStmt(SubStmt);
  }
}

void CodeGenFunction::EmitGotoStmt(const GotoStmt &S) {
  // If this code is reachable then emit a stop point (if generating
  // debug info). We have to do this ourselves because we are on the
  // "simple" statement path.
  if (HaveInsertPoint())
    EmitStopPoint(&S);

  EmitBranchThroughCleanup(getJumpDestForLabel(S.getLabel()));
}


void CodeGenFunction::EmitIndirectGotoStmt(const IndirectGotoStmt &S) {
  if (const LabelDecl *Target = S.getConstantTarget()) {
    EmitBranchThroughCleanup(getJumpDestForLabel(Target));
    return;
  }

  // Ensure that we have an i8* for our PHI node.
  llvm::Value *V = Builder.CreateBitCast(EmitScalarExpr(S.getTarget()),
                                         Int8PtrTy, "addr");
  llvm::BasicBlock *CurBB = Builder.GetInsertBlock();

  // Get the basic block for the indirect goto.
  llvm::BasicBlock *IndGotoBB = GetIndirectGotoBlock();

  // The first instruction in the block has to be the PHI for the switch dest,
  // add an entry for this branch.
  cast<llvm::PHINode>(IndGotoBB->begin())->addIncoming(V, CurBB);

  EmitBranch(IndGotoBB);
}

void CodeGenFunction::EmitIfStmt(const IfStmt &S) {
  // C99 6.8.4.1: The first substatement is executed if the expression compares
  // unequal to 0.  The condition must be a scalar type.
  LexicalScope ConditionScope(*this, S.getCond()->getSourceRange());

  if (S.getConditionVariable())
    EmitAutoVarDecl(*S.getConditionVariable());

  // If the condition constant folds and can be elided, try to avoid emitting
  // the condition and the dead arm of the if/else.
  bool CondConstant;
  if (ConstantFoldsToSimpleInteger(S.getCond(), CondConstant)) {
    // Figure out which block (then or else) is executed.
    const Stmt *Executed = S.getThen();
    const Stmt *Skipped  = S.getElse();
    if (!CondConstant)  // Condition false?
      std::swap(Executed, Skipped);

    // If the skipped block has no labels in it, just emit the executed block.
    // This avoids emitting dead code and simplifies the CFG substantially.
    if (!ContainsLabel(Skipped)) {
      if (CondConstant)
        incrementProfileCounter(&S);
      if (Executed) {
        RunCleanupsScope ExecutedScope(*this);
        EmitStmt(Executed);
      }
      return;
    }
  }

  // Otherwise, the condition did not fold, or we couldn't elide it.  Just emit
  // the conditional branch.
  llvm::BasicBlock *ThenBlock = createBasicBlock("if.then");
  llvm::BasicBlock *ContBlock = createBasicBlock("if.end");
  llvm::BasicBlock *ElseBlock = ContBlock;
  if (S.getElse())
    ElseBlock = createBasicBlock("if.else");

  EmitBranchOnBoolExpr(S.getCond(), ThenBlock, ElseBlock,
                       getProfileCount(S.getThen()));

  // Emit the 'then' code.
  EmitBlock(ThenBlock);
  incrementProfileCounter(&S);
  {
    RunCleanupsScope ThenScope(*this);
    EmitStmt(S.getThen());
  }
  EmitBranch(ContBlock);

  // Emit the 'else' code if present.
  if (const Stmt *Else = S.getElse()) {
    {
      // There is no need to emit line number for an unconditional branch.
      auto NL = ApplyDebugLocation::CreateEmpty(*this);
      EmitBlock(ElseBlock);
    }
    {
      RunCleanupsScope ElseScope(*this);
      EmitStmt(Else);
    }
    {
      // There is no need to emit line number for an unconditional branch.
      auto NL = ApplyDebugLocation::CreateEmpty(*this);
      EmitBranch(ContBlock);
    }
  }

  // Emit the continuation block for code after the if.
  EmitBlock(ContBlock, true);
}

void CodeGenFunction::EmitWhileStmt(const WhileStmt &S,
                                    ArrayRef<const Attr *> WhileAttrs) {
  // Emit the header for the loop, which will also become
  // the continue target.
  JumpDest LoopHeader = getJumpDestInCurrentScope("while.cond");
  EmitBlock(LoopHeader.getBlock());

  LoopStack.push(LoopHeader.getBlock(), CGM.getContext(), WhileAttrs);

  // Create an exit block for when the condition fails, which will
  // also become the break target.
  JumpDest LoopExit = getJumpDestInCurrentScope("while.end");

  // Store the blocks to use for break and continue.
  BreakContinueStack.push_back(BreakContinue(LoopExit, LoopHeader));

  // C++ [stmt.while]p2:
  //   When the condition of a while statement is a declaration, the
  //   scope of the variable that is declared extends from its point
  //   of declaration (3.3.2) to the end of the while statement.
  //   [...]
  //   The object created in a condition is destroyed and created
  //   with each iteration of the loop.
  RunCleanupsScope ConditionScope(*this);

  if (S.getConditionVariable())
    EmitAutoVarDecl(*S.getConditionVariable());

  // Evaluate the conditional in the while header.  C99 6.8.5.1: The
  // evaluation of the controlling expression takes place before each
  // execution of the loop body.
  llvm::Value *BoolCondVal = EvaluateExprAsBool(S.getCond());

  // while(1) is common, avoid extra exit blocks.  Be sure
  // to correctly handle break/continue though.
  bool EmitBoolCondBranch = true;
  if (llvm::ConstantInt *C = dyn_cast<llvm::ConstantInt>(BoolCondVal))
    if (C->isOne())
      EmitBoolCondBranch = false;

  // As long as the condition is true, go to the loop body.
  llvm::BasicBlock *LoopBody = createBasicBlock("while.body");
  if (EmitBoolCondBranch) {
    llvm::BasicBlock *ExitBlock = LoopExit.getBlock();
    if (ConditionScope.requiresCleanups())
      ExitBlock = createBasicBlock("while.exit");
    Builder.CreateCondBr(
        BoolCondVal, LoopBody, ExitBlock,
        createProfileWeightsForLoop(S.getCond(), getProfileCount(S.getBody())));

    if (ExitBlock != LoopExit.getBlock()) {
      EmitBlock(ExitBlock);
      EmitBranchThroughCleanup(LoopExit);
    }
  }

  // Emit the loop body.  We have to emit this in a cleanup scope
  // because it might be a singleton DeclStmt.
  {
    RunCleanupsScope BodyScope(*this);
    EmitBlock(LoopBody);
    incrementProfileCounter(&S);
    EmitStmt(S.getBody());
  }

  BreakContinueStack.pop_back();

  // Immediately force cleanup.
  ConditionScope.ForceCleanup();

  EmitStopPoint(&S);
  // Branch to the loop header again.
  EmitBranch(LoopHeader.getBlock());

  LoopStack.pop();

  // Emit the exit block.
  EmitBlock(LoopExit.getBlock(), true);

  // The LoopHeader typically is just a branch if we skipped emitting
  // a branch, try to erase it.
  if (!EmitBoolCondBranch)
    SimplifyForwardingBlocks(LoopHeader.getBlock());
}

void CodeGenFunction::EmitDoStmt(const DoStmt &S,
                                 ArrayRef<const Attr *> DoAttrs) {
  JumpDest LoopExit = getJumpDestInCurrentScope("do.end");
  JumpDest LoopCond = getJumpDestInCurrentScope("do.cond");

  uint64_t ParentCount = getCurrentProfileCount();

  // Store the blocks to use for break and continue.
  BreakContinueStack.push_back(BreakContinue(LoopExit, LoopCond));

  // Emit the body of the loop.
  llvm::BasicBlock *LoopBody = createBasicBlock("do.body");

  LoopStack.push(LoopBody, CGM.getContext(), DoAttrs);

  EmitBlockWithFallThrough(LoopBody, &S);
  {
    RunCleanupsScope BodyScope(*this);
    EmitStmt(S.getBody());
  }

  EmitBlock(LoopCond.getBlock());

  // C99 6.8.5.2: "The evaluation of the controlling expression takes place
  // after each execution of the loop body."

  // Evaluate the conditional in the while header.
  // C99 6.8.5p2/p4: The first substatement is executed if the expression
  // compares unequal to 0.  The condition must be a scalar type.
  llvm::Value *BoolCondVal = EvaluateExprAsBool(S.getCond());

  BreakContinueStack.pop_back();

  // "do {} while (0)" is common in macros, avoid extra blocks.  Be sure
  // to correctly handle break/continue though.
  bool EmitBoolCondBranch = true;
  if (llvm::ConstantInt *C = dyn_cast<llvm::ConstantInt>(BoolCondVal))
    if (C->isZero())
      EmitBoolCondBranch = false;

  // As long as the condition is true, iterate the loop.
  if (EmitBoolCondBranch) {
    uint64_t BackedgeCount = getProfileCount(S.getBody()) - ParentCount;
    Builder.CreateCondBr(
        BoolCondVal, LoopBody, LoopExit.getBlock(),
        createProfileWeightsForLoop(S.getCond(), BackedgeCount));
  }

  LoopStack.pop();

  // Emit the exit block.
  EmitBlock(LoopExit.getBlock());

  // The DoCond block typically is just a branch if we skipped
  // emitting a branch, try to erase it.
  if (!EmitBoolCondBranch)
    SimplifyForwardingBlocks(LoopCond.getBlock());
}

void CodeGenFunction::EmitForStmt(const ForStmt &S,
                                  ArrayRef<const Attr *> ForAttrs) {
  JumpDest LoopExit = getJumpDestInCurrentScope("for.end");

  LexicalScope ForScope(*this, S.getSourceRange());

  // Evaluate the first part before the loop.
  if (S.getInit())
    EmitStmt(S.getInit());

  // Start the loop with a block that tests the condition.
  // If there's an increment, the continue scope will be overwritten
  // later.
  JumpDest Continue = getJumpDestInCurrentScope("for.cond");
  llvm::BasicBlock *CondBlock = Continue.getBlock();
  EmitBlock(CondBlock);

  LoopStack.push(CondBlock, CGM.getContext(), ForAttrs);

  // If the for loop doesn't have an increment we can just use the
  // condition as the continue block.  Otherwise we'll need to create
  // a block for it (in the current scope, i.e. in the scope of the
  // condition), and that we will become our continue block.
  if (S.getInc())
    Continue = getJumpDestInCurrentScope("for.inc");

  // Store the blocks to use for break and continue.
  BreakContinueStack.push_back(BreakContinue(LoopExit, Continue));

  // Create a cleanup scope for the condition variable cleanups.
  LexicalScope ConditionScope(*this, S.getSourceRange());

  if (S.getCond()) {
    // If the for statement has a condition scope, emit the local variable
    // declaration.
    if (S.getConditionVariable()) {
      EmitAutoVarDecl(*S.getConditionVariable());
    }

    llvm::BasicBlock *ExitBlock = LoopExit.getBlock();
    // If there are any cleanups between here and the loop-exit scope,
    // create a block to stage a loop exit along.
    if (ForScope.requiresCleanups())
      ExitBlock = createBasicBlock("for.cond.cleanup");

    // As long as the condition is true, iterate the loop.
    llvm::BasicBlock *ForBody = createBasicBlock("for.body");

    // C99 6.8.5p2/p4: The first substatement is executed if the expression
    // compares unequal to 0.  The condition must be a scalar type.
    llvm::Value *BoolCondVal = EvaluateExprAsBool(S.getCond());
    Builder.CreateCondBr(
        BoolCondVal, ForBody, ExitBlock,
        createProfileWeightsForLoop(S.getCond(), getProfileCount(S.getBody())));

    if (ExitBlock != LoopExit.getBlock()) {
      EmitBlock(ExitBlock);
      EmitBranchThroughCleanup(LoopExit);
    }

    EmitBlock(ForBody);
  } else {
    // Treat it as a non-zero constant.  Don't even create a new block for the
    // body, just fall into it.
  }
  incrementProfileCounter(&S);

  {
    // Create a separate cleanup scope for the body, in case it is not
    // a compound statement.
    RunCleanupsScope BodyScope(*this);
    EmitStmt(S.getBody());
  }

  // If there is an increment, emit it next.
  if (S.getInc()) {
    EmitBlock(Continue.getBlock());
    EmitStmt(S.getInc());
  }

  BreakContinueStack.pop_back();

  ConditionScope.ForceCleanup();

  EmitStopPoint(&S);
  EmitBranch(CondBlock);

  ForScope.ForceCleanup();

  LoopStack.pop();

  // Emit the fall-through block.
  EmitBlock(LoopExit.getBlock(), true);
}

void
CodeGenFunction::EmitCXXForRangeStmt(const CXXForRangeStmt &S,
                                     ArrayRef<const Attr *> ForAttrs) {
  JumpDest LoopExit = getJumpDestInCurrentScope("for.end");

  LexicalScope ForScope(*this, S.getSourceRange());

  // Evaluate the first pieces before the loop.
  EmitStmt(S.getRangeStmt());
  EmitStmt(S.getBeginEndStmt());

  // Start the loop with a block that tests the condition.
  // If there's an increment, the continue scope will be overwritten
  // later.
  llvm::BasicBlock *CondBlock = createBasicBlock("for.cond");
  EmitBlock(CondBlock);

  LoopStack.push(CondBlock, CGM.getContext(), ForAttrs);

  // If there are any cleanups between here and the loop-exit scope,
  // create a block to stage a loop exit along.
  llvm::BasicBlock *ExitBlock = LoopExit.getBlock();
  if (ForScope.requiresCleanups())
    ExitBlock = createBasicBlock("for.cond.cleanup");

  // The loop body, consisting of the specified body and the loop variable.
  llvm::BasicBlock *ForBody = createBasicBlock("for.body");

  // The body is executed if the expression, contextually converted
  // to bool, is true.
  llvm::Value *BoolCondVal = EvaluateExprAsBool(S.getCond());
  Builder.CreateCondBr(
      BoolCondVal, ForBody, ExitBlock,
      createProfileWeightsForLoop(S.getCond(), getProfileCount(S.getBody())));

  if (ExitBlock != LoopExit.getBlock()) {
    EmitBlock(ExitBlock);
    EmitBranchThroughCleanup(LoopExit);
  }

  EmitBlock(ForBody);
  incrementProfileCounter(&S);

  // Create a block for the increment. In case of a 'continue', we jump there.
  JumpDest Continue = getJumpDestInCurrentScope("for.inc");

  // Store the blocks to use for break and continue.
  BreakContinueStack.push_back(BreakContinue(LoopExit, Continue));

  {
    // Create a separate cleanup scope for the loop variable and body.
    LexicalScope BodyScope(*this, S.getSourceRange());
    EmitStmt(S.getLoopVarStmt());
    EmitStmt(S.getBody());
  }

  EmitStopPoint(&S);
  // If there is an increment, emit it next.
  EmitBlock(Continue.getBlock());
  EmitStmt(S.getInc());

  BreakContinueStack.pop_back();

  EmitBranch(CondBlock);

  ForScope.ForceCleanup();

  LoopStack.pop();

  // Emit the fall-through block.
  EmitBlock(LoopExit.getBlock(), true);
}

void CodeGenFunction::EmitReturnOfRValue(RValue RV, QualType Ty) {
  if (RV.isScalar()) {
    Builder.CreateStore(RV.getScalarVal(), ReturnValue);
  } else if (RV.isAggregate()) {
    EmitAggregateCopy(ReturnValue, RV.getAggregateAddress(), Ty);
  } else {
    EmitStoreOfComplex(RV.getComplexVal(), MakeAddrLValue(ReturnValue, Ty),
                       /*init*/ true);
  }
  EmitBranchThroughCleanup(ReturnBlock);
}

/// EmitReturnStmt - Note that due to GCC extensions, this can have an operand
/// if the function returns void, or may be missing one if the function returns
/// non-void.  Fun stuff :).
void CodeGenFunction::EmitReturnStmt(const ReturnStmt &S) {
  // Returning from an outlined SEH helper is UB, and we already warn on it.
  if (IsOutlinedSEHHelper) {
    Builder.CreateUnreachable();
    Builder.ClearInsertionPoint();
  }

  // Emit the result value, even if unused, to evalute the side effects.
  const Expr *RV = S.getRetValue();

  // Treat block literals in a return expression as if they appeared
  // in their own scope.  This permits a small, easily-implemented
  // exception to our over-conservative rules about not jumping to
  // statements following block literals with non-trivial cleanups.
  RunCleanupsScope cleanupScope(*this);
  if (const ExprWithCleanups *cleanups =
        dyn_cast_or_null<ExprWithCleanups>(RV)) {
    enterFullExpression(cleanups);
    RV = cleanups->getSubExpr();
  }

  // FIXME: Clean this up by using an LValue for ReturnTemp,
  // EmitStoreThroughLValue, and EmitAnyExpr.
  if (getLangOpts().ElideConstructors &&
      S.getNRVOCandidate() && S.getNRVOCandidate()->isNRVOVariable()) {
    // Apply the named return value optimization for this return statement,
    // which means doing nothing: the appropriate result has already been
    // constructed into the NRVO variable.

    // If there is an NRVO flag for this variable, set it to 1 into indicate
    // that the cleanup code should not destroy the variable.
    if (llvm::Value *NRVOFlag = NRVOFlags[S.getNRVOCandidate()])
      Builder.CreateFlagStore(Builder.getTrue(), NRVOFlag);
  } else if (!ReturnValue.isValid() || (RV && RV->getType()->isVoidType())) {
    // Make sure not to return anything, but evaluate the expression
    // for side effects.
    if (RV)
      EmitAnyExpr(RV);
  } else if (!RV) {
    // Do nothing (return value is left uninitialized)
  } else if (FnRetTy->isReferenceType()) {
    // If this function returns a reference, take the address of the expression
    // rather than the value.
    RValue Result = EmitReferenceBindingToExpr(RV);
    Builder.CreateStore(Result.getScalarVal(), ReturnValue);
  } else {
    switch (getEvaluationKind(RV->getType())) {
    case TEK_Scalar:
      Builder.CreateStore(EmitScalarExpr(RV), ReturnValue);
      break;
    case TEK_Complex:
      EmitComplexExprIntoLValue(RV, MakeAddrLValue(ReturnValue, RV->getType()),
                                /*isInit*/ true);
      break;
    case TEK_Aggregate:
      EmitAggExpr(RV, AggValueSlot::forAddr(ReturnValue,
                                            Qualifiers(),
                                            AggValueSlot::IsDestructed,
                                            AggValueSlot::DoesNotNeedGCBarriers,
                                            AggValueSlot::IsNotAliased));
      break;
    }
  }

  ++NumReturnExprs;
  if (!RV || RV->isEvaluatable(getContext()))
    ++NumSimpleReturnExprs;

  cleanupScope.ForceCleanup();
  EmitBranchThroughCleanup(ReturnBlock);
}

void CodeGenFunction::EmitDeclStmt(const DeclStmt &S) {
  // As long as debug info is modeled with instructions, we have to ensure we
  // have a place to insert here and write the stop point here.
  if (HaveInsertPoint())
    EmitStopPoint(&S);

  for (const auto *I : S.decls())
    EmitDecl(*I);
}

void CodeGenFunction::EmitBreakStmt(const BreakStmt &S) {
  assert(!BreakContinueStack.empty() && "break stmt not in a loop or switch!");

  // If this code is reachable then emit a stop point (if generating
  // debug info). We have to do this ourselves because we are on the
  // "simple" statement path.
  if (HaveInsertPoint())
    EmitStopPoint(&S);

  EmitBranchThroughCleanup(BreakContinueStack.back().BreakBlock);
}

void CodeGenFunction::EmitContinueStmt(const ContinueStmt &S) {
  assert(!BreakContinueStack.empty() && "continue stmt not in a loop!");

  // If this code is reachable then emit a stop point (if generating
  // debug info). We have to do this ourselves because we are on the
  // "simple" statement path.
  if (HaveInsertPoint())
    EmitStopPoint(&S);

  EmitBranchThroughCleanup(BreakContinueStack.back().ContinueBlock);
}

/// EmitCaseStmtRange - If case statement range is not too big then
/// add multiple cases to switch instruction, one for each value within
/// the range. If range is too big then emit "if" condition check.
void CodeGenFunction::EmitCaseStmtRange(const CaseStmt &S) {
  assert(S.getRHS() && "Expected RHS value in CaseStmt");

  llvm::APSInt LHS = S.getLHS()->EvaluateKnownConstInt(getContext());
  llvm::APSInt RHS = S.getRHS()->EvaluateKnownConstInt(getContext());

  // Emit the code for this case. We do this first to make sure it is
  // properly chained from our predecessor before generating the
  // switch machinery to enter this block.
  llvm::BasicBlock *CaseDest = createBasicBlock("sw.bb");
  EmitBlockWithFallThrough(CaseDest, &S);
  EmitStmt(S.getSubStmt());

  // If range is empty, do nothing.
  if (LHS.isSigned() ? RHS.slt(LHS) : RHS.ult(LHS))
    return;

  llvm::APInt Range = RHS - LHS;
  // FIXME: parameters such as this should not be hardcoded.
  if (Range.ult(llvm::APInt(Range.getBitWidth(), 64))) {
    // Range is small enough to add multiple switch instruction cases.
    uint64_t Total = getProfileCount(&S);
    unsigned NCases = Range.getZExtValue() + 1;
    // We only have one region counter for the entire set of cases here, so we
    // need to divide the weights evenly between the generated cases, ensuring
    // that the total weight is preserved. E.g., a weight of 5 over three cases
    // will be distributed as weights of 2, 2, and 1.
    uint64_t Weight = Total / NCases, Rem = Total % NCases;
    for (unsigned I = 0; I != NCases; ++I) {
      if (SwitchWeights)
        SwitchWeights->push_back(Weight + (Rem ? 1 : 0));
      if (Rem)
        Rem--;
      SwitchInsn->addCase(Builder.getInt(LHS), CaseDest);
      LHS++;
    }
    return;
  }

  // The range is too big. Emit "if" condition into a new block,
  // making sure to save and restore the current insertion point.
  llvm::BasicBlock *RestoreBB = Builder.GetInsertBlock();

  // Push this test onto the chain of range checks (which terminates
  // in the default basic block). The switch's default will be changed
  // to the top of this chain after switch emission is complete.
  llvm::BasicBlock *FalseDest = CaseRangeBlock;
  CaseRangeBlock = createBasicBlock("sw.caserange");

  CurFn->getBasicBlockList().push_back(CaseRangeBlock);
  Builder.SetInsertPoint(CaseRangeBlock);

  // Emit range check.
  llvm::Value *Diff =
    Builder.CreateSub(SwitchInsn->getCondition(), Builder.getInt(LHS));
  llvm::Value *Cond =
    Builder.CreateICmpULE(Diff, Builder.getInt(Range), "inbounds");

  llvm::MDNode *Weights = nullptr;
  if (SwitchWeights) {
    uint64_t ThisCount = getProfileCount(&S);
    uint64_t DefaultCount = (*SwitchWeights)[0];
    Weights = createProfileWeights(ThisCount, DefaultCount);

    // Since we're chaining the switch default through each large case range, we
    // need to update the weight for the default, ie, the first case, to include
    // this case.
    (*SwitchWeights)[0] += ThisCount;
  }
  Builder.CreateCondBr(Cond, CaseDest, FalseDest, Weights);

  // Restore the appropriate insertion point.
  if (RestoreBB)
    Builder.SetInsertPoint(RestoreBB);
  else
    Builder.ClearInsertionPoint();
}

void CodeGenFunction::EmitCaseStmt(const CaseStmt &S) {
  // If there is no enclosing switch instance that we're aware of, then this
  // case statement and its block can be elided.  This situation only happens
  // when we've constant-folded the switch, are emitting the constant case,
  // and part of the constant case includes another case statement.  For
  // instance: switch (4) { case 4: do { case 5: } while (1); }
  if (!SwitchInsn) {
    EmitStmt(S.getSubStmt());
    return;
  }

  // Handle case ranges.
  if (S.getRHS()) {
    EmitCaseStmtRange(S);
    return;
  }

  llvm::ConstantInt *CaseVal =
    Builder.getInt(S.getLHS()->EvaluateKnownConstInt(getContext()));

  // If the body of the case is just a 'break', try to not emit an empty block.
  // If we're profiling or we're not optimizing, leave the block in for better
  // debug and coverage analysis.
  if (!CGM.getCodeGenOpts().ProfileInstrGenerate &&
      CGM.getCodeGenOpts().OptimizationLevel > 0 &&
      isa<BreakStmt>(S.getSubStmt())) {
    JumpDest Block = BreakContinueStack.back().BreakBlock;

    // Only do this optimization if there are no cleanups that need emitting.
    if (isObviouslyBranchWithoutCleanups(Block)) {
      if (SwitchWeights)
        SwitchWeights->push_back(getProfileCount(&S));
      SwitchInsn->addCase(CaseVal, Block.getBlock());

      // If there was a fallthrough into this case, make sure to redirect it to
      // the end of the switch as well.
      if (Builder.GetInsertBlock()) {
        Builder.CreateBr(Block.getBlock());
        Builder.ClearInsertionPoint();
      }
      return;
    }
  }

  llvm::BasicBlock *CaseDest = createBasicBlock("sw.bb");
  EmitBlockWithFallThrough(CaseDest, &S);
  if (SwitchWeights)
    SwitchWeights->push_back(getProfileCount(&S));
  SwitchInsn->addCase(CaseVal, CaseDest);

  // Recursively emitting the statement is acceptable, but is not wonderful for
  // code where we have many case statements nested together, i.e.:
  //  case 1:
  //    case 2:
  //      case 3: etc.
  // Handling this recursively will create a new block for each case statement
  // that falls through to the next case which is IR intensive.  It also causes
  // deep recursion which can run into stack depth limitations.  Handle
  // sequential non-range case statements specially.
  const CaseStmt *CurCase = &S;
  const CaseStmt *NextCase = dyn_cast<CaseStmt>(S.getSubStmt());

  // Otherwise, iteratively add consecutive cases to this switch stmt.
  while (NextCase && NextCase->getRHS() == nullptr) {
    CurCase = NextCase;
    llvm::ConstantInt *CaseVal =
      Builder.getInt(CurCase->getLHS()->EvaluateKnownConstInt(getContext()));

    if (SwitchWeights)
      SwitchWeights->push_back(getProfileCount(NextCase));
    if (CGM.getCodeGenOpts().ProfileInstrGenerate) {
      CaseDest = createBasicBlock("sw.bb");
      EmitBlockWithFallThrough(CaseDest, &S);
    }

    SwitchInsn->addCase(CaseVal, CaseDest);
    NextCase = dyn_cast<CaseStmt>(CurCase->getSubStmt());
  }

  // Normal default recursion for non-cases.
  EmitStmt(CurCase->getSubStmt());
}

void CodeGenFunction::EmitDefaultStmt(const DefaultStmt &S) {
  llvm::BasicBlock *DefaultBlock = SwitchInsn->getDefaultDest();
  assert(DefaultBlock->empty() &&
         "EmitDefaultStmt: Default block already defined?");

  EmitBlockWithFallThrough(DefaultBlock, &S);

  EmitStmt(S.getSubStmt());
}

/// CollectStatementsForCase - Given the body of a 'switch' statement and a
/// constant value that is being switched on, see if we can dead code eliminate
/// the body of the switch to a simple series of statements to emit.  Basically,
/// on a switch (5) we want to find these statements:
///    case 5:
///      printf(...);    <--
///      ++i;            <--
///      break;
///
/// and add them to the ResultStmts vector.  If it is unsafe to do this
/// transformation (for example, one of the elided statements contains a label
/// that might be jumped to), return CSFC_Failure.  If we handled it and 'S'
/// should include statements after it (e.g. the printf() line is a substmt of
/// the case) then return CSFC_FallThrough.  If we handled it and found a break
/// statement, then return CSFC_Success.
///
/// If Case is non-null, then we are looking for the specified case, checking
/// that nothing we jump over contains labels.  If Case is null, then we found
/// the case and are looking for the break.
///
/// If the recursive walk actually finds our Case, then we set FoundCase to
/// true.
///
enum CSFC_Result { CSFC_Failure, CSFC_FallThrough, CSFC_Success };
static CSFC_Result CollectStatementsForCase(const Stmt *S,
                                            const SwitchCase *Case,
                                            bool &FoundCase,
                              SmallVectorImpl<const Stmt*> &ResultStmts) {
  // If this is a null statement, just succeed.
  if (!S)
    return Case ? CSFC_Success : CSFC_FallThrough;

  // If this is the switchcase (case 4: or default) that we're looking for, then
  // we're in business.  Just add the substatement.
  if (const SwitchCase *SC = dyn_cast<SwitchCase>(S)) {
    if (S == Case) {
      FoundCase = true;
      return CollectStatementsForCase(SC->getSubStmt(), nullptr, FoundCase,
                                      ResultStmts);
    }

    // Otherwise, this is some other case or default statement, just ignore it.
    return CollectStatementsForCase(SC->getSubStmt(), Case, FoundCase,
                                    ResultStmts);
  }

  // If we are in the live part of the code and we found our break statement,
  // return a success!
  if (!Case && isa<BreakStmt>(S))
    return CSFC_Success;

  // If this is a switch statement, then it might contain the SwitchCase, the
  // break, or neither.
  if (const CompoundStmt *CS = dyn_cast<CompoundStmt>(S)) {
    // Handle this as two cases: we might be looking for the SwitchCase (if so
    // the skipped statements must be skippable) or we might already have it.
    CompoundStmt::const_body_iterator I = CS->body_begin(), E = CS->body_end();
    if (Case) {
      // Keep track of whether we see a skipped declaration.  The code could be
      // using the declaration even if it is skipped, so we can't optimize out
      // the decl if the kept statements might refer to it.
      bool HadSkippedDecl = false;

      // If we're looking for the case, just see if we can skip each of the
      // substatements.
      for (; Case && I != E; ++I) {
        HadSkippedDecl |= isa<DeclStmt>(*I);

        switch (CollectStatementsForCase(*I, Case, FoundCase, ResultStmts)) {
        case CSFC_Failure: return CSFC_Failure;
        case CSFC_Success:
          // A successful result means that either 1) that the statement doesn't
          // have the case and is skippable, or 2) does contain the case value
          // and also contains the break to exit the switch.  In the later case,
          // we just verify the rest of the statements are elidable.
          if (FoundCase) {
            // If we found the case and skipped declarations, we can't do the
            // optimization.
            if (HadSkippedDecl)
              return CSFC_Failure;

            for (++I; I != E; ++I)
              if (CodeGenFunction::ContainsLabel(*I, true))
                return CSFC_Failure;
            return CSFC_Success;
          }
          break;
        case CSFC_FallThrough:
          // If we have a fallthrough condition, then we must have found the
          // case started to include statements.  Consider the rest of the
          // statements in the compound statement as candidates for inclusion.
          assert(FoundCase && "Didn't find case but returned fallthrough?");
          // We recursively found Case, so we're not looking for it anymore.
          Case = nullptr;

          // If we found the case and skipped declarations, we can't do the
          // optimization.
          if (HadSkippedDecl)
            return CSFC_Failure;
          break;
        }
      }
    }

    // If we have statements in our range, then we know that the statements are
    // live and need to be added to the set of statements we're tracking.
    for (; I != E; ++I) {
      switch (CollectStatementsForCase(*I, nullptr, FoundCase, ResultStmts)) {
      case CSFC_Failure: return CSFC_Failure;
      case CSFC_FallThrough:
        // A fallthrough result means that the statement was simple and just
        // included in ResultStmt, keep adding them afterwards.
        break;
      case CSFC_Success:
        // A successful result means that we found the break statement and
        // stopped statement inclusion.  We just ensure that any leftover stmts
        // are skippable and return success ourselves.
        for (++I; I != E; ++I)
          if (CodeGenFunction::ContainsLabel(*I, true))
            return CSFC_Failure;
        return CSFC_Success;
      }
    }

    return Case ? CSFC_Success : CSFC_FallThrough;
  }

  // Okay, this is some other statement that we don't handle explicitly, like a
  // for statement or increment etc.  If we are skipping over this statement,
  // just verify it doesn't have labels, which would make it invalid to elide.
  if (Case) {
    if (CodeGenFunction::ContainsLabel(S, true))
      return CSFC_Failure;
    return CSFC_Success;
  }

  // Otherwise, we want to include this statement.  Everything is cool with that
  // so long as it doesn't contain a break out of the switch we're in.
  if (CodeGenFunction::containsBreak(S)) return CSFC_Failure;

  // Otherwise, everything is great.  Include the statement and tell the caller
  // that we fall through and include the next statement as well.
  ResultStmts.push_back(S);
  return CSFC_FallThrough;
}

/// FindCaseStatementsForValue - Find the case statement being jumped to and
/// then invoke CollectStatementsForCase to find the list of statements to emit
/// for a switch on constant.  See the comment above CollectStatementsForCase
/// for more details.
static bool FindCaseStatementsForValue(const SwitchStmt &S,
                                       const llvm::APSInt &ConstantCondValue,
                                SmallVectorImpl<const Stmt*> &ResultStmts,
                                       ASTContext &C,
                                       const SwitchCase *&ResultCase) {
  // First step, find the switch case that is being branched to.  We can do this
  // efficiently by scanning the SwitchCase list.
  const SwitchCase *Case = S.getSwitchCaseList();
  const DefaultStmt *DefaultCase = nullptr;

  for (; Case; Case = Case->getNextSwitchCase()) {
    // It's either a default or case.  Just remember the default statement in
    // case we're not jumping to any numbered cases.
    if (const DefaultStmt *DS = dyn_cast<DefaultStmt>(Case)) {
      DefaultCase = DS;
      continue;
    }

    // Check to see if this case is the one we're looking for.
    const CaseStmt *CS = cast<CaseStmt>(Case);
    // Don't handle case ranges yet.
    if (CS->getRHS()) return false;

    // If we found our case, remember it as 'case'.
    if (CS->getLHS()->EvaluateKnownConstInt(C) == ConstantCondValue)
      break;
  }

  // If we didn't find a matching case, we use a default if it exists, or we
  // elide the whole switch body!
  if (!Case) {
    // It is safe to elide the body of the switch if it doesn't contain labels
    // etc.  If it is safe, return successfully with an empty ResultStmts list.
    if (!DefaultCase)
      return !CodeGenFunction::ContainsLabel(&S);
    Case = DefaultCase;
  }

  // Ok, we know which case is being jumped to, try to collect all the
  // statements that follow it.  This can fail for a variety of reasons.  Also,
  // check to see that the recursive walk actually found our case statement.
  // Insane cases like this can fail to find it in the recursive walk since we
  // don't handle every stmt kind:
  // switch (4) {
  //   while (1) {
  //     case 4: ...
  bool FoundCase = false;
  ResultCase = Case;
  return CollectStatementsForCase(S.getBody(), Case, FoundCase,
                                  ResultStmts) != CSFC_Failure &&
         FoundCase;
}

void CodeGenFunction::EmitSwitchStmt(const SwitchStmt &S) {
  // Handle nested switch statements.
  llvm::SwitchInst *SavedSwitchInsn = SwitchInsn;
  SmallVector<uint64_t, 16> *SavedSwitchWeights = SwitchWeights;
  llvm::BasicBlock *SavedCRBlock = CaseRangeBlock;

  // See if we can constant fold the condition of the switch and therefore only
  // emit the live case statement (if any) of the switch.
  llvm::APSInt ConstantCondValue;
  if (ConstantFoldsToSimpleInteger(S.getCond(), ConstantCondValue)) {
    SmallVector<const Stmt*, 4> CaseStmts;
    const SwitchCase *Case = nullptr;
    if (FindCaseStatementsForValue(S, ConstantCondValue, CaseStmts,
                                   getContext(), Case)) {
      if (Case)
        incrementProfileCounter(Case);
      RunCleanupsScope ExecutedScope(*this);

      // Emit the condition variable if needed inside the entire cleanup scope
      // used by this special case for constant folded switches.
      if (S.getConditionVariable())
        EmitAutoVarDecl(*S.getConditionVariable());

      // At this point, we are no longer "within" a switch instance, so
      // we can temporarily enforce this to ensure that any embedded case
      // statements are not emitted.
      SwitchInsn = nullptr;

      // Okay, we can dead code eliminate everything except this case.  Emit the
      // specified series of statements and we're good.
      for (unsigned i = 0, e = CaseStmts.size(); i != e; ++i)
        EmitStmt(CaseStmts[i]);
      incrementProfileCounter(&S);

      // Now we want to restore the saved switch instance so that nested
      // switches continue to function properly
      SwitchInsn = SavedSwitchInsn;

      return;
    }
  }

  JumpDest SwitchExit = getJumpDestInCurrentScope("sw.epilog");

  RunCleanupsScope ConditionScope(*this);
  if (S.getConditionVariable())
    EmitAutoVarDecl(*S.getConditionVariable());
  llvm::Value *CondV = EmitScalarExpr(S.getCond());

  // Create basic block to hold stuff that comes after switch
  // statement. We also need to create a default block now so that
  // explicit case ranges tests can have a place to jump to on
  // failure.
  llvm::BasicBlock *DefaultBlock = createBasicBlock("sw.default");
  SwitchInsn = Builder.CreateSwitch(CondV, DefaultBlock);
  if (PGO.haveRegionCounts()) {
    // Walk the SwitchCase list to find how many there are.
    uint64_t DefaultCount = 0;
    unsigned NumCases = 0;
    for (const SwitchCase *Case = S.getSwitchCaseList();
         Case;
         Case = Case->getNextSwitchCase()) {
      if (isa<DefaultStmt>(Case))
        DefaultCount = getProfileCount(Case);
      NumCases += 1;
    }
    SwitchWeights = new SmallVector<uint64_t, 16>();
    SwitchWeights->reserve(NumCases);
    // The default needs to be first. We store the edge count, so we already
    // know the right weight.
    SwitchWeights->push_back(DefaultCount);
  }
  CaseRangeBlock = DefaultBlock;

  // Clear the insertion point to indicate we are in unreachable code.
  Builder.ClearInsertionPoint();

  // All break statements jump to NextBlock. If BreakContinueStack is non-empty
  // then reuse last ContinueBlock.
  JumpDest OuterContinue;
  if (!BreakContinueStack.empty())
    OuterContinue = BreakContinueStack.back().ContinueBlock;

  BreakContinueStack.push_back(BreakContinue(SwitchExit, OuterContinue));

  // Emit switch body.
  EmitStmt(S.getBody());

  BreakContinueStack.pop_back();

  // Update the default block in case explicit case range tests have
  // been chained on top.
  SwitchInsn->setDefaultDest(CaseRangeBlock);

  // If a default was never emitted:
  if (!DefaultBlock->getParent()) {
    // If we have cleanups, emit the default block so that there's a
    // place to jump through the cleanups from.
    if (ConditionScope.requiresCleanups()) {
      EmitBlock(DefaultBlock);

    // Otherwise, just forward the default block to the switch end.
    } else {
      DefaultBlock->replaceAllUsesWith(SwitchExit.getBlock());
      delete DefaultBlock;
    }
  }

  ConditionScope.ForceCleanup();

  // Emit continuation.
  EmitBlock(SwitchExit.getBlock(), true);
  incrementProfileCounter(&S);

  // If the switch has a condition wrapped by __builtin_unpredictable,
  // create metadata that specifies that the switch is unpredictable.
  // Don't bother if not optimizing because that metadata would not be used.
  if (CGM.getCodeGenOpts().OptimizationLevel != 0) {
    if (const CallExpr *Call = dyn_cast<CallExpr>(S.getCond())) {
      const Decl *TargetDecl = Call->getCalleeDecl();
      if (const FunctionDecl *FD = dyn_cast_or_null<FunctionDecl>(TargetDecl)) {
        if (FD->getBuiltinID() == Builtin::BI__builtin_unpredictable) {
          llvm::MDBuilder MDHelper(getLLVMContext());
          SwitchInsn->setMetadata(llvm::LLVMContext::MD_unpredictable,
                                  MDHelper.createUnpredictable());
        }
      }
    }
  }

  if (SwitchWeights) {
    assert(SwitchWeights->size() == 1 + SwitchInsn->getNumCases() &&
           "switch weights do not match switch cases");
    // If there's only one jump destination there's no sense weighting it.
    if (SwitchWeights->size() > 1)
      SwitchInsn->setMetadata(llvm::LLVMContext::MD_prof,
                              createProfileWeights(*SwitchWeights));
    delete SwitchWeights;
  }
  SwitchInsn = SavedSwitchInsn;
  SwitchWeights = SavedSwitchWeights;
  CaseRangeBlock = SavedCRBlock;
}

static std::string
SimplifyConstraint(const char *Constraint, const TargetInfo &Target,
                 SmallVectorImpl<TargetInfo::ConstraintInfo> *OutCons=nullptr) {
  std::string Result;

  while (*Constraint) {
    switch (*Constraint) {
    default:
      Result += Target.convertConstraint(Constraint);
      break;
    // Ignore these
    case '*':
    case '?':
    case '!':
    case '=': // Will see this and the following in mult-alt constraints.
    case '+':
      break;
    case '#': // Ignore the rest of the constraint alternative.
      while (Constraint[1] && Constraint[1] != ',')
        Constraint++;
      break;
    case '&':
    case '%':
      Result += *Constraint;
      while (Constraint[1] && Constraint[1] == *Constraint)
        Constraint++;
      break;
    case ',':
      Result += "|";
      break;
    case 'g':
      Result += "imr";
      break;
    case '[': {
      assert(OutCons &&
             "Must pass output names to constraints with a symbolic name");
      unsigned Index;
      bool result = Target.resolveSymbolicName(Constraint,
                                               &(*OutCons)[0],
                                               OutCons->size(), Index);
      assert(result && "Could not resolve symbolic name"); (void)result;
      Result += llvm::utostr(Index);
      break;
    }
    }

    Constraint++;
  }

  return Result;
}

/// AddVariableConstraints - Look at AsmExpr and if it is a variable declared
/// as using a particular register add that as a constraint that will be used
/// in this asm stmt.
static std::string
AddVariableConstraints(const std::string &Constraint, const Expr &AsmExpr,
                       const TargetInfo &Target, CodeGenModule &CGM,
                       const AsmStmt &Stmt, const bool EarlyClobber) {
  const DeclRefExpr *AsmDeclRef = dyn_cast<DeclRefExpr>(&AsmExpr);
  if (!AsmDeclRef)
    return Constraint;
  const ValueDecl &Value = *AsmDeclRef->getDecl();
  const VarDecl *Variable = dyn_cast<VarDecl>(&Value);
  if (!Variable)
    return Constraint;
  if (Variable->getStorageClass() != SC_Register)
    return Constraint;
  AsmLabelAttr *Attr = Variable->getAttr<AsmLabelAttr>();
  if (!Attr)
    return Constraint;
  StringRef Register = Attr->getLabel();
  assert(Target.isValidGCCRegisterName(Register));
  // We're using validateOutputConstraint here because we only care if
  // this is a register constraint.
  TargetInfo::ConstraintInfo Info(Constraint, "");
  if (Target.validateOutputConstraint(Info) &&
      !Info.allowsRegister()) {
    CGM.ErrorUnsupported(&Stmt, "__asm__");
    return Constraint;
  }
  // Canonicalize the register here before returning it.
  Register = Target.getNormalizedGCCRegisterName(Register);
  return (EarlyClobber ? "&{" : "{") + Register.str() + "}";
}

llvm::Value*
CodeGenFunction::EmitAsmInputLValue(const TargetInfo::ConstraintInfo &Info,
                                    LValue InputValue, QualType InputType,
                                    std::string &ConstraintStr,
                                    SourceLocation Loc) {
  llvm::Value *Arg;
  if (Info.allowsRegister() || !Info.allowsMemory()) {
    if (CodeGenFunction::hasScalarEvaluationKind(InputType)) {
      Arg = EmitLoadOfLValue(InputValue, Loc).getScalarVal();
    } else {
      llvm::Type *Ty = ConvertType(InputType);
      uint64_t Size = CGM.getDataLayout().getTypeSizeInBits(Ty);
      if (Size <= 64 && llvm::isPowerOf2_64(Size)) {
        Ty = llvm::IntegerType::get(getLLVMContext(), Size);
        Ty = llvm::PointerType::getUnqual(Ty);

        Arg = Builder.CreateLoad(Builder.CreateBitCast(InputValue.getAddress(),
                                                       Ty));
      } else {
        Arg = InputValue.getPointer();
        ConstraintStr += '*';
      }
    }
  } else {
    Arg = InputValue.getPointer();
    ConstraintStr += '*';
  }

  return Arg;
}

llvm::Value* CodeGenFunction::EmitAsmInput(
                                         const TargetInfo::ConstraintInfo &Info,
                                           const Expr *InputExpr,
                                           std::string &ConstraintStr) {
  // If this can't be a register or memory, i.e., has to be a constant
  // (immediate or symbolic), try to emit it as such.
  if (!Info.allowsRegister() && !Info.allowsMemory()) {
    llvm::APSInt Result;
    if (InputExpr->EvaluateAsInt(Result, getContext()))
      return llvm::ConstantInt::get(getLLVMContext(), Result);
    assert(!Info.requiresImmediateConstant() &&
           "Required-immediate inlineasm arg isn't constant?");
  }

  if (Info.allowsRegister() || !Info.allowsMemory())
    if (CodeGenFunction::hasScalarEvaluationKind(InputExpr->getType()))
      return EmitScalarExpr(InputExpr);

  InputExpr = InputExpr->IgnoreParenNoopCasts(getContext());
  LValue Dest = EmitLValue(InputExpr);
  return EmitAsmInputLValue(Info, Dest, InputExpr->getType(), ConstraintStr,
                            InputExpr->getExprLoc());
}

/// getAsmSrcLocInfo - Return the !srcloc metadata node to attach to an inline
/// asm call instruction.  The !srcloc MDNode contains a list of constant
/// integers which are the source locations of the start of each line in the
/// asm.
static llvm::MDNode *getAsmSrcLocInfo(const StringLiteral *Str,
                                      CodeGenFunction &CGF) {
  SmallVector<llvm::Metadata *, 8> Locs;
  // Add the location of the first line to the MDNode.
  Locs.push_back(llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(
      CGF.Int32Ty, Str->getLocStart().getRawEncoding())));
  StringRef StrVal = Str->getString();
  if (!StrVal.empty()) {
    const SourceManager &SM = CGF.CGM.getContext().getSourceManager();
    const LangOptions &LangOpts = CGF.CGM.getLangOpts();

    // Add the location of the start of each subsequent line of the asm to the
    // MDNode.
    for (unsigned i = 0, e = StrVal.size()-1; i != e; ++i) {
      if (StrVal[i] != '\n') continue;
      SourceLocation LineLoc = Str->getLocationOfByte(i+1, SM, LangOpts,
                                                      CGF.getTarget());
      Locs.push_back(llvm::ConstantAsMetadata::get(
          llvm::ConstantInt::get(CGF.Int32Ty, LineLoc.getRawEncoding())));
    }
  }

  return llvm::MDNode::get(CGF.getLLVMContext(), Locs);
}

void CodeGenFunction::EmitAsmStmt(const AsmStmt &S) {
  // Assemble the final asm string.
  std::string AsmString = S.generateAsmString(getContext());

  // Get all the output and input constraints together.
  SmallVector<TargetInfo::ConstraintInfo, 4> OutputConstraintInfos;
  SmallVector<TargetInfo::ConstraintInfo, 4> InputConstraintInfos;

  for (unsigned i = 0, e = S.getNumOutputs(); i != e; i++) {
    StringRef Name;
    if (const GCCAsmStmt *GAS = dyn_cast<GCCAsmStmt>(&S))
      Name = GAS->getOutputName(i);
    TargetInfo::ConstraintInfo Info(S.getOutputConstraint(i), Name);
    bool IsValid = getTarget().validateOutputConstraint(Info); (void)IsValid;
    assert(IsValid && "Failed to parse output constraint");
    OutputConstraintInfos.push_back(Info);
  }

  for (unsigned i = 0, e = S.getNumInputs(); i != e; i++) {
    StringRef Name;
    if (const GCCAsmStmt *GAS = dyn_cast<GCCAsmStmt>(&S))
      Name = GAS->getInputName(i);
    TargetInfo::ConstraintInfo Info(S.getInputConstraint(i), Name);
    bool IsValid =
      getTarget().validateInputConstraint(OutputConstraintInfos.data(),
                                          S.getNumOutputs(), Info);
    assert(IsValid && "Failed to parse input constraint"); (void)IsValid;
    InputConstraintInfos.push_back(Info);
  }

  std::string Constraints;

  std::vector<LValue> ResultRegDests;
  std::vector<QualType> ResultRegQualTys;
  std::vector<llvm::Type *> ResultRegTypes;
  std::vector<llvm::Type *> ResultTruncRegTypes;
  std::vector<llvm::Type *> ArgTypes;
  std::vector<llvm::Value*> Args;

  // Keep track of inout constraints.
  std::string InOutConstraints;
  std::vector<llvm::Value*> InOutArgs;
  std::vector<llvm::Type*> InOutArgTypes;

  // An inline asm can be marked readonly if it meets the following conditions:
  //  - it doesn't have any sideeffects
  //  - it doesn't clobber memory
  //  - it doesn't return a value by-reference
  // It can be marked readnone if it doesn't have any input memory constraints
  // in addition to meeting the conditions listed above.
  bool ReadOnly = true, ReadNone = true;

  for (unsigned i = 0, e = S.getNumOutputs(); i != e; i++) {
    TargetInfo::ConstraintInfo &Info = OutputConstraintInfos[i];

    // Simplify the output constraint.
    std::string OutputConstraint(S.getOutputConstraint(i));
    OutputConstraint = SimplifyConstraint(OutputConstraint.c_str() + 1,
                                          getTarget());

    const Expr *OutExpr = S.getOutputExpr(i);
    OutExpr = OutExpr->IgnoreParenNoopCasts(getContext());

    OutputConstraint = AddVariableConstraints(OutputConstraint, *OutExpr,
                                              getTarget(), CGM, S,
                                              Info.earlyClobber());

    LValue Dest = EmitLValue(OutExpr);
    if (!Constraints.empty())
      Constraints += ',';

    // If this is a register output, then make the inline asm return it
    // by-value.  If this is a memory result, return the value by-reference.
    if (!Info.allowsMemory() && hasScalarEvaluationKind(OutExpr->getType())) {
      Constraints += "=" + OutputConstraint;
      ResultRegQualTys.push_back(OutExpr->getType());
      ResultRegDests.push_back(Dest);
      ResultRegTypes.push_back(ConvertTypeForMem(OutExpr->getType()));
      ResultTruncRegTypes.push_back(ResultRegTypes.back());

      // If this output is tied to an input, and if the input is larger, then
      // we need to set the actual result type of the inline asm node to be the
      // same as the input type.
      if (Info.hasMatchingInput()) {
        unsigned InputNo;
        for (InputNo = 0; InputNo != S.getNumInputs(); ++InputNo) {
          TargetInfo::ConstraintInfo &Input = InputConstraintInfos[InputNo];
          if (Input.hasTiedOperand() && Input.getTiedOperand() == i)
            break;
        }
        assert(InputNo != S.getNumInputs() && "Didn't find matching input!");

        QualType InputTy = S.getInputExpr(InputNo)->getType();
        QualType OutputType = OutExpr->getType();

        uint64_t InputSize = getContext().getTypeSize(InputTy);
        if (getContext().getTypeSize(OutputType) < InputSize) {
          // Form the asm to return the value as a larger integer or fp type.
          ResultRegTypes.back() = ConvertType(InputTy);
        }
      }
      if (llvm::Type* AdjTy =
            getTargetHooks().adjustInlineAsmType(*this, OutputConstraint,
                                                 ResultRegTypes.back()))
        ResultRegTypes.back() = AdjTy;
      else {
        CGM.getDiags().Report(S.getAsmLoc(),
                              diag::err_asm_invalid_type_in_input)
            << OutExpr->getType() << OutputConstraint;
      }
    } else {
      ArgTypes.push_back(Dest.getAddress().getType());
      Args.push_back(Dest.getPointer());
      Constraints += "=*";
      Constraints += OutputConstraint;
      ReadOnly = ReadNone = false;
    }

    if (Info.isReadWrite()) {
      InOutConstraints += ',';

      const Expr *InputExpr = S.getOutputExpr(i);
      llvm::Value *Arg = EmitAsmInputLValue(Info, Dest, InputExpr->getType(),
                                            InOutConstraints,
                                            InputExpr->getExprLoc());

      if (llvm::Type* AdjTy =
          getTargetHooks().adjustInlineAsmType(*this, OutputConstraint,
                                               Arg->getType()))
        Arg = Builder.CreateBitCast(Arg, AdjTy);

      if (Info.allowsRegister())
        InOutConstraints += llvm::utostr(i);
      else
        InOutConstraints += OutputConstraint;

      InOutArgTypes.push_back(Arg->getType());
      InOutArgs.push_back(Arg);
    }
  }

  // If this is a Microsoft-style asm blob, store the return registers (EAX:EDX)
  // to the return value slot. Only do this when returning in registers.
  if (isa<MSAsmStmt>(&S)) {
    const ABIArgInfo &RetAI = CurFnInfo->getReturnInfo();
    if (RetAI.isDirect() || RetAI.isExtend()) {
      // Make a fake lvalue for the return value slot.
      LValue ReturnSlot = MakeAddrLValue(ReturnValue, FnRetTy);
      CGM.getTargetCodeGenInfo().addReturnRegisterOutputs(
          *this, ReturnSlot, Constraints, ResultRegTypes, ResultTruncRegTypes,
          ResultRegDests, AsmString, S.getNumOutputs());
      SawAsmBlock = true;
    }
  }

  for (unsigned i = 0, e = S.getNumInputs(); i != e; i++) {
    const Expr *InputExpr = S.getInputExpr(i);

    TargetInfo::ConstraintInfo &Info = InputConstraintInfos[i];

    if (Info.allowsMemory())
      ReadNone = false;

    if (!Constraints.empty())
      Constraints += ',';

    // Simplify the input constraint.
    std::string InputConstraint(S.getInputConstraint(i));
    InputConstraint = SimplifyConstraint(InputConstraint.c_str(), getTarget(),
                                         &OutputConstraintInfos);

    InputConstraint = AddVariableConstraints(
        InputConstraint, *InputExpr->IgnoreParenNoopCasts(getContext()),
        getTarget(), CGM, S, false /* No EarlyClobber */);

    llvm::Value *Arg = EmitAsmInput(Info, InputExpr, Constraints);

    // If this input argument is tied to a larger output result, extend the
    // input to be the same size as the output.  The LLVM backend wants to see
    // the input and output of a matching constraint be the same size.  Note
    // that GCC does not define what the top bits are here.  We use zext because
    // that is usually cheaper, but LLVM IR should really get an anyext someday.
    if (Info.hasTiedOperand()) {
      unsigned Output = Info.getTiedOperand();
      QualType OutputType = S.getOutputExpr(Output)->getType();
      QualType InputTy = InputExpr->getType();

      if (getContext().getTypeSize(OutputType) >
          getContext().getTypeSize(InputTy)) {
        // Use ptrtoint as appropriate so that we can do our extension.
        if (isa<llvm::PointerType>(Arg->getType()))
          Arg = Builder.CreatePtrToInt(Arg, IntPtrTy);
        llvm::Type *OutputTy = ConvertType(OutputType);
        if (isa<llvm::IntegerType>(OutputTy))
          Arg = Builder.CreateZExt(Arg, OutputTy);
        else if (isa<llvm::PointerType>(OutputTy))
          Arg = Builder.CreateZExt(Arg, IntPtrTy);
        else {
          assert(OutputTy->isFloatingPointTy() && "Unexpected output type");
          Arg = Builder.CreateFPExt(Arg, OutputTy);
        }
      }
    }
    if (llvm::Type* AdjTy =
              getTargetHooks().adjustInlineAsmType(*this, InputConstraint,
                                                   Arg->getType()))
      Arg = Builder.CreateBitCast(Arg, AdjTy);
    else
      CGM.getDiags().Report(S.getAsmLoc(), diag::err_asm_invalid_type_in_input)
          << InputExpr->getType() << InputConstraint;

    ArgTypes.push_back(Arg->getType());
    Args.push_back(Arg);
    Constraints += InputConstraint;
  }

  // Append the "input" part of inout constraints last.
  for (unsigned i = 0, e = InOutArgs.size(); i != e; i++) {
    ArgTypes.push_back(InOutArgTypes[i]);
    Args.push_back(InOutArgs[i]);
  }
  Constraints += InOutConstraints;

  // Clobbers
  for (unsigned i = 0, e = S.getNumClobbers(); i != e; i++) {
    StringRef Clobber = S.getClobber(i);

    if (Clobber == "memory")
      ReadOnly = ReadNone = false;
    else if (Clobber != "cc")
      Clobber = getTarget().getNormalizedGCCRegisterName(Clobber);

    if (!Constraints.empty())
      Constraints += ',';

    Constraints += "~{";
    Constraints += Clobber;
    Constraints += '}';
  }

  // Add machine specific clobbers
  std::string MachineClobbers = getTarget().getClobbers();
  if (!MachineClobbers.empty()) {
    if (!Constraints.empty())
      Constraints += ',';
    Constraints += MachineClobbers;
  }

  llvm::Type *ResultType;
  if (ResultRegTypes.empty())
    ResultType = VoidTy;
  else if (ResultRegTypes.size() == 1)
    ResultType = ResultRegTypes[0];
  else
    ResultType = llvm::StructType::get(getLLVMContext(), ResultRegTypes);

  llvm::FunctionType *FTy =
    llvm::FunctionType::get(ResultType, ArgTypes, false);

  bool HasSideEffect = S.isVolatile() || S.getNumOutputs() == 0;
  llvm::InlineAsm::AsmDialect AsmDialect = isa<MSAsmStmt>(&S) ?
    llvm::InlineAsm::AD_Intel : llvm::InlineAsm::AD_ATT;
  llvm::InlineAsm *IA =
    llvm::InlineAsm::get(FTy, AsmString, Constraints, HasSideEffect,
                         /* IsAlignStack */ false, AsmDialect);
  llvm::CallInst *Result = Builder.CreateCall(IA, Args);
  Result->addAttribute(llvm::AttributeSet::FunctionIndex,
                       llvm::Attribute::NoUnwind);

  // Attach readnone and readonly attributes.
  if (!HasSideEffect) {
    if (ReadNone)
      Result->addAttribute(llvm::AttributeSet::FunctionIndex,
                           llvm::Attribute::ReadNone);
    else if (ReadOnly)
      Result->addAttribute(llvm::AttributeSet::FunctionIndex,
                           llvm::Attribute::ReadOnly);
  }

  // Slap the source location of the inline asm into a !srcloc metadata on the
  // call.
  if (const GCCAsmStmt *gccAsmStmt = dyn_cast<GCCAsmStmt>(&S)) {
    Result->setMetadata("srcloc", getAsmSrcLocInfo(gccAsmStmt->getAsmString(),
                                                   *this));
  } else {
    // At least put the line number on MS inline asm blobs.
    auto Loc = llvm::ConstantInt::get(Int32Ty, S.getAsmLoc().getRawEncoding());
    Result->setMetadata("srcloc",
                        llvm::MDNode::get(getLLVMContext(),
                                          llvm::ConstantAsMetadata::get(Loc)));
  }

  // Extract all of the register value results from the asm.
  std::vector<llvm::Value*> RegResults;
  if (ResultRegTypes.size() == 1) {
    RegResults.push_back(Result);
  } else {
    for (unsigned i = 0, e = ResultRegTypes.size(); i != e; ++i) {
      llvm::Value *Tmp = Builder.CreateExtractValue(Result, i, "asmresult");
      RegResults.push_back(Tmp);
    }
  }

  assert(RegResults.size() == ResultRegTypes.size());
  assert(RegResults.size() == ResultTruncRegTypes.size());
  assert(RegResults.size() == ResultRegDests.size());
  for (unsigned i = 0, e = RegResults.size(); i != e; ++i) {
    llvm::Value *Tmp = RegResults[i];

    // If the result type of the LLVM IR asm doesn't match the result type of
    // the expression, do the conversion.
    if (ResultRegTypes[i] != ResultTruncRegTypes[i]) {
      llvm::Type *TruncTy = ResultTruncRegTypes[i];

      // Truncate the integer result to the right size, note that TruncTy can be
      // a pointer.
      if (TruncTy->isFloatingPointTy())
        Tmp = Builder.CreateFPTrunc(Tmp, TruncTy);
      else if (TruncTy->isPointerTy() && Tmp->getType()->isIntegerTy()) {
        uint64_t ResSize = CGM.getDataLayout().getTypeSizeInBits(TruncTy);
        Tmp = Builder.CreateTrunc(Tmp,
                   llvm::IntegerType::get(getLLVMContext(), (unsigned)ResSize));
        Tmp = Builder.CreateIntToPtr(Tmp, TruncTy);
      } else if (Tmp->getType()->isPointerTy() && TruncTy->isIntegerTy()) {
        uint64_t TmpSize =CGM.getDataLayout().getTypeSizeInBits(Tmp->getType());
        Tmp = Builder.CreatePtrToInt(Tmp,
                   llvm::IntegerType::get(getLLVMContext(), (unsigned)TmpSize));
        Tmp = Builder.CreateTrunc(Tmp, TruncTy);
      } else if (TruncTy->isIntegerTy()) {
        Tmp = Builder.CreateTrunc(Tmp, TruncTy);
      } else if (TruncTy->isVectorTy()) {
        Tmp = Builder.CreateBitCast(Tmp, TruncTy);
      }
    }

    EmitStoreThroughLValue(RValue::get(Tmp), ResultRegDests[i]);
  }
}

LValue CodeGenFunction::InitCapturedStruct(const CapturedStmt &S) {
  const RecordDecl *RD = S.getCapturedRecordDecl();
  QualType RecordTy = getContext().getRecordType(RD);

  // Initialize the captured struct.
  LValue SlotLV =
    MakeAddrLValue(CreateMemTemp(RecordTy, "agg.captured"), RecordTy);

  RecordDecl::field_iterator CurField = RD->field_begin();
  for (CapturedStmt::const_capture_init_iterator I = S.capture_init_begin(),
                                                 E = S.capture_init_end();
       I != E; ++I, ++CurField) {
    LValue LV = EmitLValueForFieldInitialization(SlotLV, *CurField);
    if (CurField->hasCapturedVLAType()) {
      auto VAT = CurField->getCapturedVLAType();
      EmitStoreThroughLValue(RValue::get(VLASizeMap[VAT->getSizeExpr()]), LV);
    } else {
      EmitInitializerForField(*CurField, LV, *I, None);
    }
  }

  return SlotLV;
}

/// Generate an outlined function for the body of a CapturedStmt, store any
/// captured variables into the captured struct, and call the outlined function.
llvm::Function *
CodeGenFunction::EmitCapturedStmt(const CapturedStmt &S, CapturedRegionKind K) {
  LValue CapStruct = InitCapturedStruct(S);

  // Emit the CapturedDecl
  CodeGenFunction CGF(CGM, true);
  CGCapturedStmtRAII CapInfoRAII(CGF, new CGCapturedStmtInfo(S, K));
  llvm::Function *F = CGF.GenerateCapturedStmtFunction(S);
  delete CGF.CapturedStmtInfo;

  // Emit call to the helper function.
  EmitCallOrInvoke(F, CapStruct.getPointer());

  return F;
}

Address CodeGenFunction::GenerateCapturedStmtArgument(const CapturedStmt &S) {
  LValue CapStruct = InitCapturedStruct(S);
  return CapStruct.getAddress();
}

/// Creates the outlined function for a CapturedStmt.
llvm::Function *
CodeGenFunction::GenerateCapturedStmtFunction(const CapturedStmt &S) {
  assert(CapturedStmtInfo &&
    "CapturedStmtInfo should be set when generating the captured function");
  const CapturedDecl *CD = S.getCapturedDecl();
  const RecordDecl *RD = S.getCapturedRecordDecl();
  SourceLocation Loc = S.getLocStart();
  assert(CD->hasBody() && "missing CapturedDecl body");

  // Build the argument list.
  ASTContext &Ctx = CGM.getContext();
  FunctionArgList Args;
  Args.append(CD->param_begin(), CD->param_end());

  // Create the function declaration.
  FunctionType::ExtInfo ExtInfo;
  const CGFunctionInfo &FuncInfo =
      CGM.getTypes().arrangeFreeFunctionDeclaration(Ctx.VoidTy, Args, ExtInfo,
                                                    /*IsVariadic=*/false);
  llvm::FunctionType *FuncLLVMTy = CGM.getTypes().GetFunctionType(FuncInfo);

  llvm::Function *F =
    llvm::Function::Create(FuncLLVMTy, llvm::GlobalValue::InternalLinkage,
                           CapturedStmtInfo->getHelperName(), &CGM.getModule());
  CGM.SetInternalFunctionAttributes(CD, F, FuncInfo);
  if (CD->isNothrow())
    F->addFnAttr(llvm::Attribute::NoUnwind);

  // Generate the function.
  StartFunction(CD, Ctx.VoidTy, F, FuncInfo, Args,
                CD->getLocation(),
                CD->getBody()->getLocStart());
  // Set the context parameter in CapturedStmtInfo.
  Address DeclPtr = GetAddrOfLocalVar(CD->getContextParam());
  CapturedStmtInfo->setContextValue(Builder.CreateLoad(DeclPtr));

  // Initialize variable-length arrays.
  LValue Base = MakeNaturalAlignAddrLValue(CapturedStmtInfo->getContextValue(),
                                           Ctx.getTagDeclType(RD));
  for (auto *FD : RD->fields()) {
    if (FD->hasCapturedVLAType()) {
      auto *ExprArg = EmitLoadOfLValue(EmitLValueForField(Base, FD),
                                       S.getLocStart()).getScalarVal();
      auto VAT = FD->getCapturedVLAType();
      VLASizeMap[VAT->getSizeExpr()] = ExprArg;
    }
  }

  // If 'this' is captured, load it into CXXThisValue.
  if (CapturedStmtInfo->isCXXThisExprCaptured()) {
    FieldDecl *FD = CapturedStmtInfo->getThisFieldDecl();
    LValue ThisLValue = EmitLValueForField(Base, FD);
    CXXThisValue = EmitLoadOfLValue(ThisLValue, Loc).getScalarVal();
  }

  PGO.assignRegionCounters(CD, F);
  CapturedStmtInfo->EmitBody(*this, CD->getBody());
  FinishFunction(CD->getBodyRBrace());

  return F;
}
