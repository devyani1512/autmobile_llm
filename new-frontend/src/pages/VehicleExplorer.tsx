import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { ArrowLeft, Info } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";

interface Hotspot {
  id: string;
  x: number;
  y: number;
  label: string;
  description: string;
  tips: string[];
}

const hotspots: Hotspot[] = [
  {
    id: "engine",
    x: 30,
    y: 45,
    label: "Engine",
    description: "The heart of your vehicle, converting fuel into mechanical energy.",
    tips: ["Check oil levels monthly", "Regular maintenance every 5,000 miles", "Listen for unusual sounds"]
  },
  {
    id: "transmission",
    x: 45,
    y: 52,
    label: "Transmission",
    description: "Transfers power from the engine to the wheels.",
    tips: ["Check transmission fluid regularly", "Smooth shifting is key", "Service every 30,000 miles"]
  },
  {
    id: "brakes",
    x: 70,
    y: 60,
    label: "Brake System",
    description: "Critical safety system for stopping your vehicle.",
    tips: ["Replace pads when worn", "Check brake fluid monthly", "Listen for squeaking sounds"]
  },
  {
    id: "battery",
    x: 25,
    y: 35,
    label: "Battery",
    description: "Powers electrical systems and starts the engine.",
    tips: ["Clean terminals regularly", "Replace every 3-5 years", "Check voltage periodically"]
  }
];

const VehicleExplorer = () => {
  const navigate = useNavigate();
  const [selectedHotspot, setSelectedHotspot] = useState<Hotspot | null>(null);

  return (
    <div className="min-h-screen bg-gradient-hero">
      {/* Header */}
      <header className="border-b border-border glass-card">
        <div className="container mx-auto px-4 py-6">
          <div className="flex items-center justify-between">
            <Button
              variant="ghost"
              onClick={() => navigate("/dashboard/owner")}
              className="gap-2"
            >
              <ArrowLeft className="w-4 h-4" />
              Back to Dashboard
            </Button>
            <h1 className="text-xl font-bold text-gradient-owner">Vehicle Explorer</h1>
            <div className="w-24" />
          </div>
        </div>
      </header>

      {/* Main Content */}
      <div className="container mx-auto px-4 py-12">
        <div className="grid md:grid-cols-2 gap-8 max-w-7xl mx-auto">
          {/* Vehicle Visualization */}
          <motion.div
            initial={{ opacity: 0, x: -30 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.6 }}
            className="relative"
          >
            <Card className="glass-card p-8">
              <h2 className="text-2xl font-bold mb-6">Interactive Vehicle Guide</h2>
              
              {/* Car SVG with hotspots */}
              <div className="relative w-full aspect-video bg-muted/20 rounded-lg overflow-hidden">
                <svg className="w-full h-full" viewBox="0 0 100 100" preserveAspectRatio="xMidYMid meet">
                  {/* Car outline */}
                  <path
                    d="M 15 45 L 20 45 L 22 40 L 30 40 L 32 35 L 68 35 L 70 40 L 78 40 L 80 45 L 85 45 L 85 55 L 82 55 C 82 60, 78 65, 73 65 C 68 65, 64 60, 64 55 L 36 55 C 36 60, 32 65, 27 65 C 22 65, 18 60, 18 55 L 15 55 Z"
                    fill="hsl(0 0% 14%)"
                    stroke="hsl(210 100% 50%)"
                    strokeWidth="0.5"
                  />
                  
                  {/* Hotspots */}
                  {hotspots.map((hotspot) => (
                    <g key={hotspot.id}>
                      <motion.circle
                        cx={hotspot.x}
                        cy={hotspot.y}
                        r="2"
                        fill="hsl(210 100% 50%)"
                        className="cursor-pointer"
                        onClick={() => setSelectedHotspot(hotspot)}
                        whileHover={{ scale: 1.5 }}
                        animate={{
                          opacity: [0.6, 1, 0.6],
                          scale: [1, 1.2, 1]
                        }}
                        transition={{
                          duration: 2,
                          repeat: Infinity,
                          ease: "easeInOut"
                        }}
                      />
                      <circle
                        cx={hotspot.x}
                        cy={hotspot.y}
                        r="3"
                        fill="none"
                        stroke="hsl(210 100% 50%)"
                        strokeWidth="0.3"
                        opacity="0.5"
                        className="pointer-events-none"
                      />
                    </g>
                  ))}
                </svg>
              </div>

              <p className="text-sm text-muted-foreground mt-4 text-center">
                Click on the glowing points to learn more about each component
              </p>
            </Card>
          </motion.div>

          {/* Info Panel */}
          <motion.div
            initial={{ opacity: 0, x: 30 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.6 }}
          >
            <AnimatePresence mode="wait">
              {selectedHotspot ? (
                <motion.div
                  key={selectedHotspot.id}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -20 }}
                  transition={{ duration: 0.3 }}
                >
                  <Card className="glass-card p-8 border-2 border-accent-owner/50 shadow-glow-owner">
                    <div className="flex items-start gap-4 mb-6">
                      <div className="w-12 h-12 rounded-lg bg-gradient-owner flex items-center justify-center">
                        <Info className="w-6 h-6 text-white" />
                      </div>
                      <div>
                        <h3 className="text-2xl font-bold text-gradient-owner">{selectedHotspot.label}</h3>
                        <p className="text-muted-foreground mt-2">{selectedHotspot.description}</p>
                      </div>
                    </div>

                    <div className="mb-6">
                      <h4 className="text-lg font-semibold mb-3">Maintenance Tips</h4>
                      <ul className="space-y-2">
                        {selectedHotspot.tips.map((tip, index) => (
                          <li key={index} className="flex items-start gap-2">
                            <div className="w-1.5 h-1.5 rounded-full bg-accent-owner mt-2" />
                            <span className="text-muted-foreground">{tip}</span>
                          </li>
                        ))}
                      </ul>
                    </div>

                    <Button
                      onClick={() => navigate("/chat?mode=owner&component=" + selectedHotspot.id)}
                      className="w-full bg-gradient-owner hover:opacity-90 text-white"
                    >
                      Ask AI About This Component
                    </Button>
                  </Card>
                </motion.div>
              ) : (
                <motion.div
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  exit={{ opacity: 0 }}
                >
                  <Card className="glass-card p-8 h-full flex items-center justify-center">
                    <div className="text-center">
                      <div className="w-16 h-16 rounded-full bg-gradient-owner/20 flex items-center justify-center mx-auto mb-4">
                        <Info className="w-8 h-8 text-accent-owner" />
                      </div>
                      <p className="text-muted-foreground">
                        Select a component to learn more
                      </p>
                    </div>
                  </Card>
                </motion.div>
              )}
            </AnimatePresence>
          </motion.div>
        </div>
      </div>
    </div>
  );
};

export default VehicleExplorer;
