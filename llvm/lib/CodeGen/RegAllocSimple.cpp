//===----------------------------------------------------------------------===//
//
/// A register allocator simplified from RegAllocFast.cpp
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SparseSet.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineOperand.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/RegAllocRegistry.h"
#include "llvm/CodeGen/Register.h"
#include "llvm/CodeGen/RegisterClassInfo.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"
#include "llvm/CodeGen/VirtRegMap.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"

#include "llvm/CodeGen/LiveIntervals.h"
#include "llvm/CodeGen/SlotIndexes.h"

#include <algorithm>
#include <cstdint>
#include <iterator>
#include <map>
#include <numeric>
#include <set>

using namespace llvm;

#define DEBUG_TYPE "regalloc"

STATISTIC(NumStores, "Number of stores added");
STATISTIC(NumLoads , "Number of loads added");

namespace {
  /// This is class where you will implement your register allocator in
  class RegAllocSimple : public MachineFunctionPass {
  public:
    static char ID;
    RegAllocSimple() : MachineFunctionPass(ID), SpillMap(-1) {}

  private:
    /// Some information that might be useful for register allocation
    /// They are initialized in runOnMachineFunction
    MachineFrameInfo *MFI;
    MachineRegisterInfo *MRI;
    const TargetRegisterInfo *TRI;
    const TargetInstrInfo *TII;
    RegisterClassInfo RegClassInfo;

    // TODO: maintain information about live registers
    using RegisterSet = SmallSet<Register, 8>;
    IndexedMap<Register, VirtReg2IndexFunctor> LiveVirtRegs;
    RegisterSet LivePhysRegs;
    IndexedMap<int, VirtReg2IndexFunctor> SpillMap;

  public:
    StringRef getPassName() const override { return "Simple Register Allocator"; }

    void getAnalysisUsage(AnalysisUsage &AU) const override {
      AU.setPreservesCFG();
      // At -O1/-O2, llc fails to schedule some required passes if this pass
      // does not preserve these anlyses; these are preserved by recomputing
      // them at the end of runOnFunction(), you can safely ignore these
      AU.addRequired<LiveIntervals>();
      AU.addPreserved<LiveIntervals>();
      AU.addRequired<SlotIndexes>();
      AU.addPreserved<SlotIndexes>();
      MachineFunctionPass::getAnalysisUsage(AU);
    }

    /// Ask the Machine IR verifier to check some simple properties
    /// Enabled with the -verify-machineinstrs flag in llc
    MachineFunctionProperties getRequiredProperties() const override {
      return MachineFunctionProperties().set(
          MachineFunctionProperties::Property::NoPHIs);
    }

    MachineFunctionProperties getSetProperties() const override {
      return MachineFunctionProperties().set(
          MachineFunctionProperties::Property::NoVRegs);
    }

    MachineFunctionProperties getClearedProperties() const override {
      return MachineFunctionProperties().set(
        MachineFunctionProperties::Property::IsSSA);
    }

  private:
    int allocateStackSlot(Register Reg) {
      if (SpillMap[Reg] != -1) {
        return SpillMap[Reg];
      }

      const TargetRegisterClass *TRC = MRI->getRegClass(Reg);
      int StackSlot = MFI->CreateSpillStackObject(TRI->getSpillSize(*TRC),
                                                  TRI->getSpillAlign(*TRC));
      SpillMap[Reg] = StackSlot;
      return StackSlot;
    }

    Register findAvailablePhysicalRegister(Register VirtReg) {
      // TODO: Naive implementation won't be fast
      for (Register PhysReg : RegClassInfo.getOrder(MRI->getRegClass(VirtReg))) {
        bool PhysRegIsAvailable = true;
        for (const Register &LivePhysReg : LivePhysRegs) {
          if (TRI->regsOverlap(LivePhysReg, PhysReg)) {
            PhysRegIsAvailable = false;
            break;
          }
        }
        for (unsigned VirtRegIndex = 0;
             PhysRegIsAvailable && VirtRegIndex < LiveVirtRegs.size();
             VirtRegIndex++) {
          Register VirtReg = Register::index2VirtReg(VirtRegIndex);
          // XXX: Also need to check that the registers are not aliased (eg.
          // subregisters)
          if (TRI->regsOverlap(LiveVirtRegs[VirtReg], PhysReg)) {
            PhysRegIsAvailable = false;
            break;
          }
        }
        if (PhysRegIsAvailable) {
          return PhysReg;
        }
      }
      return Register();
    }

    /// Allocate physical register for virtual register operand
    Register allocateOperand(MachineOperand &MO, MachineBasicBlock::iterator MI,
                             bool IsUse, RegisterSet &UsedRegisters) {
      // TODO: allocate physical register for a virtual register
      Register Reg = MO.getReg();
      MachineBasicBlock *MBB = MO.getParent()->getParent();
      const TargetRegisterClass *TRC = MRI->getRegClass(Reg);
      if (LiveVirtRegs[Reg] != Register()) {
        return LiveVirtRegs[Reg];
      }

      Register PhysReg = findAvailablePhysicalRegister(Reg);
      if (PhysReg == Register()) {
        bool SuccessfullySpilledRegister = false;
        // TODO: Would be nice to have an iterator over IndexedMap<Register, Virt2Phys...>
        for (unsigned VirtRegIndex = 0; VirtRegIndex < LiveVirtRegs.size(); VirtRegIndex++) {
          Register VirtReg = Register::index2VirtReg(VirtRegIndex);
          PhysReg = LiveVirtRegs[VirtReg];
          if (PhysReg != Register() && !MO.getParent()->readsVirtualRegister(VirtReg)) {
            // TODO: What should IsKill be here?
            int StackSlot = allocateStackSlot(PhysReg);
            TII->storeRegToStackSlot(*MBB, MI, PhysReg, false, StackSlot, TRC, TRI);
            LiveVirtRegs[VirtReg] = Register();
            LivePhysRegs.erase(PhysReg);
            SuccessfullySpilledRegister = true;
            break;
          }
        }
        assert(SuccessfullySpilledRegister && "Could not find an available physical register");
      }

      if (MO.isUse()) {
        int StackSlot = allocateStackSlot(Reg);
        TII->loadRegFromStackSlot(*MBB, MI, PhysReg, StackSlot, TRC, TRI);
      }

      LiveVirtRegs[Reg] = PhysReg;
      return PhysReg;
    }

    void allocateInstruction(MachineInstr &MI, MachineBasicBlock::iterator MIt) {
      // XXX: find and allocate all virtual registers in MI
      RegisterSet UsedInInstr;
      // TODO: Would be nice to do this in a single loop, but need to make sure to visit uses before defs
      SmallVector<unsigned> OperandIndexes(MI.getNumOperands());
      std::iota(OperandIndexes.begin(), OperandIndexes.end(), 0);
      std::sort(OperandIndexes.begin(), OperandIndexes.end(),
                [&](int A, int B) {
                  MachineOperand MOA = MI.getOperand(A), MOB = MI.getOperand(B);
                  auto Score =
                      [&](MachineOperand &MO) {
                        return (MO.isRegMask() << 2) |
                          // Sorting by physicality probably doesn't matter
                          ((MO.isReg() && MO.getReg().isPhysical()) << 1) | 
                          (MO.isReg() && MO.isUse());
                      };
                  uint8_t AVal = Score(MOA);
                  uint8_t BVal = Score(MOB);
                  return AVal > BVal;
                });
      for (unsigned Index : OperandIndexes) {
        MachineOperand &MO = MI.getOperand(Index);
        // TODO: I hypothesize that I am having some issues when an
        // MachineInstruction has physical register operands after virtual
        // register operands. In this case the virtual operands may be assigned
        // to a register already in use by a physical register operand.
        if (MO.isRegMask()) {
          for (unsigned RegIndex = 0; RegIndex < LiveVirtRegs.size();
               RegIndex++) {
            Register VirtReg = Register::index2VirtReg(RegIndex);
            Register PhysReg = LiveVirtRegs[VirtReg];
            if (PhysReg != Register() &&
                MO.clobbersPhysReg(PhysReg) // &&
                // !UsedInInstr.contains(VirtReg)
                ) {

              const TargetRegisterClass *TRC = MRI->getRegClass(VirtReg);
              int StackSlot = allocateStackSlot(VirtReg);
              // TODO: What should isKill be here? (Note: it is apparently possible to find uses of virtual registers by looking at MRI)
              TII->storeRegToStackSlot(*MI.getParent(), MIt, PhysReg, false, StackSlot, TRC, TRI);
              LiveVirtRegs[VirtReg] = Register();
            }
          }
        }

        if (MO.isReg() && MO.getReg().isPhysical()) {
        // TODO: I think UsedInInstr needs to be physical registers. It's current use may be derivable from the MI, but not certain. NBD if I need an additional set of virtual registers (I don't think I should because LiveVirtRegs should be sufficient?), but I definetly need usedininstr to keep track of physicla registers and I need to check every current use of it to fix it. Hm, this might also be the time to consider live ins as well. (Note: I staged the changs before I started tiredly trying to address this)
          UsedInInstr.insert(MO.getReg());

          if (MO.isDef()) {
            for (unsigned RegIndex = 0; RegIndex < LiveVirtRegs.size();
                 RegIndex++) {
              Register VirtReg = Register::index2VirtReg(RegIndex);
              Register PhysReg = LiveVirtRegs[VirtReg];
              if (TRI->regsOverlap(PhysReg, MO.getReg())) {
                assert(!UsedInInstr.contains(VirtReg) && "Cannot clear room for physical register because we already assigned a virtual register to it in the same instruction");
                const TargetRegisterClass *TRC = MRI->getRegClass(VirtReg);
                int StackSlot = allocateStackSlot(VirtReg);
                // TODO: What should isKill be here? (Note: it is apparently
                // possible to find uses of virtual registers by looking at MRI)
                TII->storeRegToStackSlot(*MI.getParent(), MIt, PhysReg, false,
                                         StackSlot, TRC, TRI);
                LiveVirtRegs[VirtReg] = Register();
              }
            }
          }

          if (MO.isDead()) {
            LivePhysRegs.erase(MO.getReg());
          } else {
            LivePhysRegs.insert(MO.getReg());
          }
        }
        if (MO.isReg() && MO.getReg().isVirtual()) {
          Register OriginalVirtualRegister = MO.getReg();
          Register AllocatedRegister =
              allocateOperand(MO, MIt, MO.isUse(), UsedInInstr);
          UsedInInstr.insert(AllocatedRegister);
          MO.substPhysReg(AllocatedRegister, *TRI);

          if (MO.isDead()) {
            LiveVirtRegs[OriginalVirtualRegister] = Register();
          }
        }
      }
      // for (MachineOperand &MO : MI.defs()) {
      //   if (MO.isReg() && MO.getReg().isVirtual()) {
      //     Register OriginalVirtualRegister = MO.getReg();
      //     Register AllocatedRegister =
      //       allocateOperand(MO, MIt, MO.isUse(), UsedInInstr);
      //     UsedInInstr.insert(MO.getReg());
      //     MO.substPhysReg(AllocatedRegister, *TRI);

      //     if (MO.isDead()) {
      //       LiveVirtRegs[OriginalVirtualRegister] = Register();
      //     }
      //   }
      // }
    }

    void allocateBasicBlock(MachineBasicBlock &MBB) {
      // XXX: allocate each instruction
      // XXX: spill all live registers at the end
      LiveVirtRegs.clear();
      LiveVirtRegs.resize(MBB.getParent()->getRegInfo().getNumVirtRegs());

      for (MachineBasicBlock::iterator MIt = MBB.begin(); MIt != MBB.end(); MIt++) {
        allocateInstruction(*MIt, MIt);
      }

      MachineBasicBlock::iterator MI = MBB.getFirstTerminator();
      if (MI != MBB.end() && MI->isReturn()) {
        return;
      }
      // if (!MI->isBranch()) { MI = std::next(MI); }
      
      for (unsigned RegIndex = 0; RegIndex < LiveVirtRegs.size();
           RegIndex++) {
        Register Reg = Register::index2VirtReg(RegIndex);
        if (LiveVirtRegs[Reg] != Register()) {
          // TODO: What should IsKill be here?
          // TODO: Should LiveVirtRegs map operands somehow?
          int StackSlot = allocateStackSlot(Reg);
          TII->storeRegToStackSlot(MBB, MI, LiveVirtRegs[Reg], false,
                                   StackSlot, MRI->getRegClass(Reg), TRI);
        }
      }
    }

    bool runOnMachineFunction(MachineFunction &MF) override {
      dbgs() << "simple regalloc running on: " << MF.getName() << "\n";

      // Get some useful information about the target
      MRI = &MF.getRegInfo();
      const TargetSubtargetInfo &STI = MF.getSubtarget();
      TRI = STI.getRegisterInfo();
      TII = STI.getInstrInfo();
      MFI = &MF.getFrameInfo();
      MRI->freezeReservedRegs(MF);
      RegClassInfo.runOnMachineFunction(MF);

      SpillMap.clear();
      SpillMap.resize(MF.getRegInfo().getNumVirtRegs());

      // Allocate each basic block locally
      for (MachineBasicBlock &MBB : MF) {
        allocateBasicBlock(MBB);
      }

      MRI->clearVirtRegs();

      // Recompute the analyses that we marked as preserved above, you can
      // safely ignore this code
      SlotIndexes& SI = getAnalysis<SlotIndexes>();
      SI.releaseMemory();
      SI.runOnMachineFunction(MF);

      LiveIntervals& LI = getAnalysis<LiveIntervals>();
      LI.releaseMemory();
      LI.runOnMachineFunction(MF);

      return true;
    }
  };
}

/// Create the initializer and register the pass
char RegAllocSimple::ID = 0;
FunctionPass *llvm::createSimpleRegisterAllocator() { return new RegAllocSimple(); }
INITIALIZE_PASS(RegAllocSimple, "regallocsimple", "Simple Register Allocator", false, false)
static RegisterRegAlloc simpleRegAlloc("simple", "simple register allocator", createSimpleRegisterAllocator);
