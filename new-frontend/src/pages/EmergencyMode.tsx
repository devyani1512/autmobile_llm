import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { ArrowLeft, AlertTriangle, Flame, Wrench, Battery, AlertCircle, Car, Phone } from "lucide-react";
import { motion } from "framer-motion";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog";

interface Emergency {
  id: string;
  title: string;
  icon: any;
  severity: "critical" | "high" | "medium";
  steps: string[];
  warnings: string[];
}

const emergencies: Emergency[] = [
  {
    id: "overheating",
    title: "Engine Overheating",
    icon: Flame,
    severity: "critical",
    steps: [
      "Pull over safely and turn off the engine immediately",
      "Do NOT open the radiator cap while hot",
      "Wait 15-20 minutes for engine to cool",
      "Check coolant level once cool",
      "Add coolant if low (50/50 mix with water)",
      "If problem persists, call for towing"
    ],
    warnings: ["Never open radiator cap when engine is hot", "Do not continue driving with overheating engine"]
  },
  {
    id: "flat-tire",
    title: "Flat Tire",
    icon: AlertCircle,
    severity: "medium",
    steps: [
      "Safely pull over to flat, stable ground",
      "Turn on hazard lights",
      "Apply parking brake",
      "Place wheel chocks",
      "Loosen lug nuts before jacking",
      "Jack up vehicle at designated points",
      "Remove flat tire and install spare",
      "Tighten lug nuts in star pattern",
      "Lower vehicle and final tighten"
    ],
    warnings: ["Never work under a vehicle supported only by a jack", "Visit mechanic to repair/replace tire"]
  },
  {
    id: "check-engine",
    title: "Check Engine Light",
    icon: AlertTriangle,
    severity: "high",
    steps: [
      "Note if light is solid or flashing",
      "Check gas cap is tight",
      "Reduce speed and avoid heavy acceleration",
      "Schedule diagnostic scan",
      "If flashing, stop driving immediately"
    ],
    warnings: ["Flashing light indicates serious issue - stop driving", "Get diagnostics within 24 hours"]
  },
  {
    id: "brake-failure",
    title: "Brake Failure",
    icon: Car,
    severity: "critical",
    steps: [
      "DO NOT PANIC - stay calm",
      "Pump brake pedal rapidly",
      "Downshift to lower gear",
      "Apply emergency/parking brake gradually",
      "Look for escape route",
      "Turn on hazard lights",
      "Once stopped, call for assistance"
    ],
    warnings: ["This is a critical emergency", "Do not restart driving - call tow truck"]
  },
  {
    id: "dead-battery",
    title: "Dead Battery",
    icon: Battery,
    severity: "medium",
    steps: [
      "Position helper vehicle close (not touching)",
      "Turn off both vehicles",
      "Connect red (+) to dead battery positive",
      "Connect red (+) to helper battery positive",
      "Connect black (-) to helper battery negative",
      "Connect black (-) to dead car metal ground",
      "Start helper vehicle, wait 2-3 minutes",
      "Try starting dead vehicle",
      "Remove cables in reverse order"
    ],
    warnings: ["Never let clamps touch each other", "Wear eye protection", "Check battery is not frozen/damaged"]
  },
  {
    id: "accident",
    title: "Accident Protocol",
    icon: Phone,
    severity: "critical",
    steps: [
      "Check for injuries - call 911 if needed",
      "Move to safe location if possible",
      "Turn on hazard lights",
      "Call police to file report",
      "Exchange information with other driver",
      "Take photos of damage and scene",
      "Document witness information",
      "Contact your insurance company"
    ],
    warnings: ["Always call police for accident report", "Never admit fault at scene"]
  }
];

const EmergencyMode = () => {
  const navigate = useNavigate();
  const [selectedEmergency, setSelectedEmergency] = useState<Emergency | null>(null);

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case "critical":
        return "from-emergency to-emergency-light";
      case "high":
        return "from-accent-mechanic to-accent-mechanic-light";
      case "medium":
        return "from-accent-owner to-accent-owner-light";
      default:
        return "from-muted to-muted-foreground";
    }
  };

  return (
    <div className="min-h-screen bg-gradient-hero">
      
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
            <div className="flex items-center gap-2">
              <AlertTriangle className="w-5 h-5 text-emergency animate-pulse" />
              <h1 className="text-xl font-bold text-emergency">Emergency Support</h1>
            </div>
            <div className="w-24" />
          </div>
        </div>
      </header>

     
      <div className="container mx-auto px-4 py-12">
        <motion.div
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
          className="text-center mb-12"
        >
          <h2 className="text-4xl md:text-5xl font-bold mb-4">Quick Access Emergency Guide</h2>
          <p className="text-xl text-muted-foreground">
            Select an emergency scenario for step-by-step instructions
          </p>
        </motion.div>

        
        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6 max-w-7xl mx-auto">
          {emergencies.map((emergency, index) => {
            const Icon = emergency.icon;
            return (
              <motion.div
                key={emergency.id}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: index * 0.1, duration: 0.5 }}
                whileHover={{ y: -5 }}
              >
                <Card
                  className="glass-card p-6 cursor-pointer border-2 border-transparent hover:border-emergency/50 transition-all duration-300 group h-full"
                  onClick={() => setSelectedEmergency(emergency)}
                >
                  <div className={`w-14 h-14 rounded-xl bg-gradient-${emergency.severity === 'critical' ? 'emergency' : emergency.severity === 'high' ? 'mechanic' : 'owner'} flex items-center justify-center mb-4 group-hover:scale-110 transition-transform duration-300`}>
                    <Icon className="w-7 h-7 text-white" />
                  </div>
                  
                  <h3 className="text-xl font-bold mb-2">{emergency.title}</h3>
                  
                  <div className="flex items-center gap-2">
                    <span className={`text-xs px-3 py-1 rounded-full bg-gradient-${getSeverityColor(emergency.severity)} text-white font-medium`}>
                      {emergency.severity.toUpperCase()}
                    </span>
                  </div>
                  
                  <Button variant="outline" className="w-full mt-4 group-hover:bg-emergency group-hover:text-white group-hover:border-emergency transition-all">
                    View Solution
                  </Button>
                </Card>
              </motion.div>
            );
          })}
        </div>
      </div>

      
      <Dialog open={!!selectedEmergency} onOpenChange={() => setSelectedEmergency(null)}>
        <DialogContent className="max-w-2xl glass-card border-emergency/50">
          {selectedEmergency && (
            <>
              <DialogHeader>
                <DialogTitle className="flex items-center gap-3 text-2xl">
                  {(() => {
                    const Icon = selectedEmergency.icon;
                    return <Icon className="w-8 h-8 text-emergency" />;
                  })()}
                  {selectedEmergency.title}
                </DialogTitle>
              </DialogHeader>

              <div className="mt-4 space-y-6">
                
                {selectedEmergency.warnings.length > 0 && (
                  <div className="bg-emergency/10 border border-emergency/30 rounded-lg p-4">
                    <h4 className="font-bold text-emergency mb-2 flex items-center gap-2">
                      <AlertTriangle className="w-5 h-5" />
                      Safety Warnings
                    </h4>
                    <ul className="space-y-1">
                      {selectedEmergency.warnings.map((warning, index) => (
                        <li key={index} className="text-sm text-muted-foreground flex items-start gap-2">
                          <span className="text-emergency">⚠</span>
                          {warning}
                        </li>
                      ))}
                    </ul>
                  </div>
                )}

                
                <div>
                  <h4 className="font-bold text-lg mb-3">Step-by-Step Instructions</h4>
                  <ol className="space-y-3">
                    {selectedEmergency.steps.map((step, index) => (
                      <li key={index} className="flex gap-3">
                        <span className="flex-shrink-0 w-8 h-8 rounded-full bg-gradient-owner flex items-center justify-center text-white font-bold text-sm">
                          {index + 1}
                        </span>
                        <span className="text-muted-foreground pt-1">{step}</span>
                      </li>
                    ))}
                  </ol>
                </div>

               
                {selectedEmergency.severity === "critical" && (
                  <div className="bg-emergency/5 border border-emergency/30 rounded-lg p-4">
                    <Button className="w-full bg-gradient-emergency hover:opacity-90 text-white text-lg py-6">
                      <Phone className="w-5 h-5 mr-2" />
                      Call Emergency Services
                    </Button>
                  </div>
                )}
              </div>
            </>
          )}
        </DialogContent>
      </Dialog>
    </div>
  );
};

export default EmergencyMode;
