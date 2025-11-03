import { useState } from "react";
import { askQuestion, uploadPDF } from "./services/api";
import { Toaster as Sonner } from "@/components/ui/sonner";
import { TooltipProvider } from "@/components/ui/tooltip";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter, Routes, Route, useLocation } from "react-router-dom";
import { AnimatePresence, motion } from "framer-motion";
import HeroLanding from "./pages/HeroLanding";
import BrandSelection from "./pages/BrandSelection";
import ModelSelection from "./pages/ModelSelection";
import ModeSelection from "./pages/ModeSelection";
import OwnerDashboard from "./pages/OwnerDashboard";
import MechanicDashboard from "./pages/MechanicDashboard";
import Chat from "./pages/Chat";
import VehicleExplorer from "./pages/VehicleExplorer";
import EmergencyMode from "./pages/EmergencyMode";
import ReportGeneration from "./pages/ReportGeneration";
import NotFound from "./pages/NotFound";
import { Toaster } from "sonner";

const queryClient = new QueryClient();

const pageVariants = {
  initial: { opacity: 0, y: 20 },
  animate: { opacity: 1, y: 0 },
  exit: { opacity: 0, y: -20 },
};

const pageTransition = {
  type: "tween" as const,
  ease: "easeInOut" as const,
  duration: 0.5,
};

const AnimatedRoutes = () => {
  const location = useLocation();

  return (
    <AnimatePresence mode="wait">
      <Routes location={location} key={location.pathname}>
        <Route
          path="/"
          element={
            <motion.div
              initial="initial"
              animate="animate"
              exit="exit"
              variants={pageVariants}
              transition={pageTransition}
            >
              <HeroLanding />
            </motion.div>
          }
        />
        <Route
          path="/brand-selection"
          element={
            <motion.div
              initial="initial"
              animate="animate"
              exit="exit"
              variants={pageVariants}
              transition={pageTransition}
            >
              <BrandSelection />
            </motion.div>
          }
        />
        <Route
          path="/model-selection"
          element={
            <motion.div
              initial="initial"
              animate="animate"
              exit="exit"
              variants={pageVariants}
              transition={pageTransition}
            >
              <ModelSelection />
            </motion.div>
          }
        />
        <Route
          path="/mode-selection"
          element={
            <motion.div
              initial="initial"
              animate="animate"
              exit="exit"
              variants={pageVariants}
              transition={pageTransition}
            >
              <ModeSelection />
            </motion.div>
          }
        />
        <Route
          path="/dashboard/owner"
          element={
            <motion.div
              initial="initial"
              animate="animate"
              exit="exit"
              variants={pageVariants}
              transition={pageTransition}
            >
              <OwnerDashboard />
            </motion.div>
          }
        />
        <Route
          path="/dashboard/mechanic"
          element={
            <motion.div
              initial="initial"
              animate="animate"
              exit="exit"
              variants={pageVariants}
              transition={pageTransition}
            >
              <MechanicDashboard />
            </motion.div>
          }
        />
        <Route
          path="/chat"
          element={
            <motion.div
              initial="initial"
              animate="animate"
              exit="exit"
              variants={pageVariants}
              transition={pageTransition}
            >
              <Chat />
            </motion.div>
          }
        />
        <Route
          path="/vehicle-explorer"
          element={
            <motion.div
              initial="initial"
              animate="animate"
              exit="exit"
              variants={pageVariants}
              transition={pageTransition}
            >
              <VehicleExplorer />
            </motion.div>
          }
        />
        <Route
          path="/emergency"
          element={
            <motion.div
              initial="initial"
              animate="animate"
              exit="exit"
              variants={pageVariants}
              transition={pageTransition}
            >
              <EmergencyMode />
            </motion.div>
          }
        />
        <Route
          path="/report-generation"
          element={
            <motion.div
              initial="initial"
              animate="animate"
              exit="exit"
              variants={pageVariants}
              transition={pageTransition}
            >
              <ReportGeneration />
            </motion.div>
          }
        />
        <Route path="*" element={<NotFound />} />
      </Routes>
    </AnimatePresence>
  );
};

const App = () => (
  <QueryClientProvider client={queryClient}>
    <TooltipProvider>
      <Toaster richColors position="top-center" />
      <Sonner />
      <BrowserRouter>
        <AnimatedRoutes />
      </BrowserRouter>
    </TooltipProvider>
  </QueryClientProvider>
);

export default App;
