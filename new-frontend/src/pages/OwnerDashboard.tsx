import { useNavigate, useSearchParams } from "react-router-dom";
import { MessageSquare, BookOpen, AlertTriangle, ArrowLeft } from "lucide-react";
import { motion } from "framer-motion";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";

const OwnerDashboard = () => {
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();
  const brand = searchParams.get("brand");
  const model = searchParams.get("model");

  const cardVariants = {
    hidden: { opacity: 0, y: 40 },
    visible: (i: number) => ({
      opacity: 1,
      y: 0,
      transition: {
        delay: i * 0.15,
        duration: 0.6,
        ease: [0.16, 1, 0.3, 1] as const
      }
    })
  };

  return (
    <div className="min-h-screen bg-gradient-hero relative overflow-hidden">
      {/* Background elements */}
      <div className="absolute inset-0 opacity-20">
        <div className="absolute top-1/3 left-1/4 w-96 h-96 bg-secondary/30 rounded-full blur-3xl animate-float" />
        <div className="absolute bottom-1/3 right-1/4 w-96 h-96 bg-chrome/20 rounded-full blur-3xl animate-float" style={{ animationDelay: "2s" }} />
      </div>

      {/* Header */}
      <header className="border-b border-border glass-card relative z-10">
        <div className="container mx-auto px-4 py-6">
          <div className="flex items-center justify-between">
            <Button
              variant="ghost"
              onClick={() => navigate(`/mode-selection?brand=${brand}&model=${model}`)}
              className="gap-2 hover:bg-secondary/10"
            >
              <ArrowLeft className="w-4 h-4" />
              Change Mode
            </Button>
            <div className="text-sm text-muted-foreground">
              Home <span className="text-foreground mx-2">›</span> {brand} {model} <span className="text-foreground mx-2">›</span> Owner Mode
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
            <span className="text-foreground">Welcome,</span>
            <br />
            <span className="text-gradient-silver">Vehicle Owner</span>
          </h1>
          <p className="text-xl text-muted-foreground font-light">
            Your personal automotive intelligence assistant
          </p>
        </motion.div>

        {/* Feature Cards */}
        <div className="grid md:grid-cols-3 gap-10 max-w-7xl mx-auto">
          {/* Ask Query Card */}
          <motion.div
            custom={0}
            variants={cardVariants}
            initial="hidden"
            animate="visible"
            whileHover={{ y: -15, scale: 1.03 }}
            onClick={() => navigate(`/chat?mode=owner&brand=${brand}&model=${model}`)}
            className="cursor-pointer"
          >
            <Card className="relative overflow-hidden h-full p-10 glass-card border-2 border-transparent hover:border-secondary/50 transition-all duration-500 group">
              <div className="absolute inset-0 bg-gradient-silver opacity-0 group-hover:opacity-10 transition-opacity duration-500" />
              <div className="absolute inset-0 shadow-glow-silver opacity-0 group-hover:opacity-100 transition-opacity duration-500" />
              
              <div className="relative z-10 text-center">
                <div className="w-20 h-20 rounded-2xl bg-gradient-silver flex items-center justify-center mb-8 mx-auto group-hover:scale-110 transition-transform duration-500 shadow-elegant">
                  <MessageSquare className="w-10 h-10 text-background" />
                </div>
                
                <h3 className="text-3xl font-bold mb-4 uppercase tracking-wide">Ask Technical Questions</h3>
                <p className="text-muted-foreground mb-8 leading-relaxed font-light">
                  Get instant expert answers about maintenance, features, and troubleshooting
                </p>
                
                <Button className="w-full bg-gradient-silver hover:bg-gradient-chrome text-background font-semibold py-6 text-lg">
                  Start Conversation
                </Button>
              </div>
            </Card>
          </motion.div>

          {/* Understand Vehicle Card */}
          <motion.div
            custom={1}
            variants={cardVariants}
            initial="hidden"
            animate="visible"
            whileHover={{ y: -15, scale: 1.03 }}
            onClick={() => navigate(`/vehicle-explorer?brand=${brand}&model=${model}`)}
            className="cursor-pointer"
          >
            <Card className="relative overflow-hidden h-full p-10 glass-card border-2 border-transparent hover:border-chrome/50 transition-all duration-500 group">
              <div className="absolute inset-0 bg-gradient-chrome opacity-0 group-hover:opacity-10 transition-opacity duration-500" />
              <div className="absolute inset-0 shadow-glow-chrome opacity-0 group-hover:opacity-100 transition-opacity duration-500" />
              
              <div className="relative z-10 text-center">
                <div className="w-20 h-20 rounded-2xl bg-gradient-chrome flex items-center justify-center mb-8 mx-auto group-hover:scale-110 transition-transform duration-500 shadow-elegant">
                  <BookOpen className="w-10 h-10 text-background" />
                </div>
                
                <h3 className="text-3xl font-bold mb-4 uppercase tracking-wide">Explore Your Vehicle</h3>
                <p className="text-muted-foreground mb-8 leading-relaxed font-light">
                  Interactive guide to every component and feature of your automobile
                </p>
                
                <Button variant="outline" className="w-full border-2 border-chrome hover:bg-chrome hover:text-background font-semibold py-6 text-lg transition-all duration-300">
                  Learn More
                </Button>
              </div>
            </Card>
          </motion.div>

          {/* Emergency Card */}
          <motion.div
            custom={2}
            variants={cardVariants}
            initial="hidden"
            animate="visible"
            whileHover={{ y: -15, scale: 1.03 }}
            onClick={() => navigate(`/emergency?brand=${brand}&model=${model}`)}
            className="cursor-pointer"
          >
            <Card className="relative overflow-hidden h-full p-10 glass-card border-2 border-transparent hover:border-emergency/50 transition-all duration-500 group">
              <div className="absolute inset-0 bg-gradient-emergency opacity-0 group-hover:opacity-15 transition-opacity duration-500" />
              <div className="absolute inset-0 shadow-glow-emergency opacity-0 group-hover:opacity-100 transition-opacity duration-500" />
              
              <div className="relative z-10 text-center">
                <div className="w-20 h-20 rounded-2xl bg-gradient-emergency flex items-center justify-center mb-8 mx-auto group-hover:scale-110 transition-transform duration-500 shadow-elegant">
                  <AlertTriangle className="w-10 h-10 text-white animate-pulse" />
                </div>
                
                <h3 className="text-3xl font-bold mb-4 uppercase tracking-wide">Emergency Support</h3>
                <p className="text-muted-foreground mb-8 leading-relaxed font-light">
                  Instant access to critical troubleshooting and safety procedures
                </p>
                
                <Button className="w-full bg-gradient-emergency hover:opacity-90 text-white font-semibold py-6 text-lg">
                  Access Emergency Guide
                </Button>
              </div>
            </Card>
          </motion.div>
        </div>
      </div>
    </div>
  );
};

export default OwnerDashboard;
