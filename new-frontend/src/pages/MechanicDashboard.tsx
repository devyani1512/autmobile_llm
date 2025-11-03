import { useNavigate, useSearchParams } from "react-router-dom";
import { Search, FileText, ArrowLeft } from "lucide-react";
import { motion } from "framer-motion";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";

const MechanicDashboard = () => {
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();
  const brand = searchParams.get("brand");
  const model = searchParams.get("model");

  const cardVariants = {
    hidden: { opacity: 0, y: 30 },
    visible: (i: number) => ({
      opacity: 1,
      y: 0,
      transition: {
        delay: i * 0.15,
        duration: 0.5,
        ease: "easeOut" as const
      }
    })
  };

  return (
    <div className="min-h-screen bg-gradient-hero relative overflow-hidden">
      {/* Background elements */}
      <div className="absolute inset-0 opacity-20">
        <div className="absolute top-1/3 left-1/3 w-96 h-96 bg-accent-mechanic/30 rounded-full blur-3xl animate-float" />
        <div className="absolute bottom-1/3 right-1/3 w-96 h-96 bg-secondary/20 rounded-full blur-3xl animate-float" style={{ animationDelay: "2s" }} />
      </div>

      {/* Header */}
      <header className="border-b border-border glass-card relative z-10">
        <div className="container mx-auto px-4 py-6">
          <div className="flex items-center justify-between">
            <Button
              variant="ghost"
              onClick={() => navigate(`/mode-selection?brand=${brand}&model=${model}`)}
              className="gap-2 hover:bg-accent-mechanic/10"
            >
              <ArrowLeft className="w-4 h-4" />
              Change Mode
            </Button>
            <div className="text-sm text-muted-foreground">
              Home <span className="text-foreground mx-2">›</span> {brand} {model} <span className="text-foreground mx-2">›</span> Mechanic Mode
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <div className="container mx-auto px-4 py-16 relative z-10">
        <motion.div
          initial={{ opacity: 0, y: -30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
          className="text-center mb-16"
        >
          <h1 className="text-5xl md:text-7xl font-bold mb-6">
            <span className="text-gradient-gold uppercase tracking-wider">
              Professional Workshop
            </span>
          </h1>
          <p className="text-xl text-muted-foreground font-light">
            Advanced diagnostic and reporting tools for professionals
          </p>
        </motion.div>

        {/* Feature Cards */}
        <div className="grid md:grid-cols-2 gap-12 max-w-6xl mx-auto">
          {/* Diagnose Problem Card */}
          <motion.div
            custom={0}
            variants={cardVariants}
            initial="hidden"
            animate="visible"
            whileHover={{ y: -15, scale: 1.03 }}
            onClick={() => navigate(`/chat?mode=mechanic&brand=${brand}&model=${model}`)}
            className="cursor-pointer"
          >
            <Card className="relative overflow-hidden h-full p-12 glass-card border-2 border-transparent hover:border-accent-mechanic/50 transition-all duration-500 group">
              <div className="absolute inset-0 bg-gradient-gold opacity-0 group-hover:opacity-15 transition-opacity duration-500" />
              <div className="absolute inset-0 shadow-glow-gold opacity-0 group-hover:opacity-100 transition-opacity duration-500" />
              
              <div className="relative z-10 text-center">
                <div className="w-24 h-24 rounded-2xl bg-gradient-gold flex items-center justify-center mb-8 mx-auto group-hover:scale-110 transition-transform duration-500 shadow-elegant">
                  <Search className="w-12 h-12 text-background" />
                </div>
                
                <h3 className="text-4xl font-bold mb-6 uppercase tracking-wide">Advanced Diagnostics</h3>
                <p className="text-muted-foreground mb-8 leading-relaxed text-lg font-light">
                  AI-powered troubleshooting with technical specifications and repair procedures. Access comprehensive diagnostic information and detailed repair instructions.
                </p>
                
                <Button className="w-full bg-gradient-gold hover:opacity-90 text-background font-bold py-6 text-lg uppercase">
                  Start Diagnosis
                </Button>
              </div>
            </Card>
          </motion.div>

          {/* Generate Report Card */}
          <motion.div
            custom={1}
            variants={cardVariants}
            initial="hidden"
            animate="visible"
            whileHover={{ y: -15, scale: 1.03 }}
            onClick={() => navigate(`/report-generation?brand=${brand}&model=${model}`)}
            className="cursor-pointer"
          >
            <Card className="relative overflow-hidden h-full p-12 glass-card border-2 border-transparent hover:border-secondary/50 transition-all duration-500 group">
              <div className="absolute inset-0 bg-gradient-silver opacity-0 group-hover:opacity-10 transition-opacity duration-500" />
              <div className="absolute inset-0 shadow-glow-silver opacity-0 group-hover:opacity-100 transition-opacity duration-500" />
              
              <div className="relative z-10 text-center">
                <div className="w-24 h-24 rounded-2xl bg-gradient-silver flex items-center justify-center mb-8 mx-auto group-hover:scale-110 transition-transform duration-500 shadow-elegant">
                  <FileText className="w-12 h-12 text-background" />
                </div>
                
                <h3 className="text-4xl font-bold mb-6 uppercase tracking-wide">Service Reports</h3>
                <p className="text-muted-foreground mb-8 leading-relaxed text-lg font-light">
                  Create comprehensive diagnostic and service reports with detailed analysis. Professional formatting ready for client delivery.
                </p>
                
                <Button variant="outline" className="w-full border-2 border-secondary hover:bg-secondary hover:text-background font-bold py-6 text-lg uppercase transition-all duration-300">
                  Create Report
                </Button>
              </div>
            </Card>
          </motion.div>
        </div>
      </div>
    </div>
  );
};

export default MechanicDashboard;
