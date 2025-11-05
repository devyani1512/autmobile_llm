import { useNavigate, useSearchParams } from "react-router-dom";
import { ArrowLeft, User, Wrench, Sparkles } from "lucide-react";
import { motion } from "framer-motion";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";

const ModeSelection = () => {
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();
  const brand = searchParams.get("brand");
  const model = searchParams.get("model");

  const cardVariants = {
    hidden: { opacity: 0, y: 50 },
    visible: (i: number) => ({
      opacity: 1,
      y: 0,
      transition: {
        delay: i * 0.2,
        duration: 0.8,
        ease: [0.16, 1, 0.3, 1] as const
      }
    })
  };

  return (
    <div className="min-h-screen bg-gradient-hero relative overflow-hidden">
      {/* Animated particles */}
      <div className="absolute inset-0 opacity-20">
        <div className="absolute top-1/4 left-1/4 w-96 h-96 bg-secondary/30 rounded-full blur-3xl animate-float" />
        <div className="absolute bottom-1/3 right-1/4 w-96 h-96 bg-accent-mechanic/30 rounded-full blur-3xl animate-float" style={{ animationDelay: "1.5s" }} />
      </div>

      {/* Header */}
      <header className="border-b border-border glass-card relative z-10">
        <div className="container mx-auto px-4 py-6">
          <div className="flex items-center justify-between">
            <Button
              variant="ghost"
              onClick={() => navigate(`/model-selection?brand=${brand}`)}
              className="gap-2 hover:bg-secondary/10"
            >
              <ArrowLeft className="w-4 h-4" />
              Change Model
            </Button>
            <div className="text-sm text-muted-foreground">
              {brand && model && (
                <>
                  Home <span className="text-foreground mx-2">›</span> 
                  {brand} <span className="text-foreground mx-2">›</span> 
                  {model} <span className="text-foreground mx-2">›</span> 
                  Mode Selection
                </>
              )}
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
          className="text-center mb-20"
        >
          <div className="inline-flex items-center gap-2 px-6 py-3 rounded-full glass-card mb-8 border border-secondary/20">
            <Sparkles className="w-5 h-5 text-secondary animate-pulse" />
            <span className="text-sm font-medium uppercase tracking-wider">Select Your Experience</span>
          </div>
          
          <h1 className="text-5xl md:text-7xl font-bold mb-6 leading-tight">
            <span className="text-foreground">Choose Your</span>
            <br />
            <span className="text-gradient-chrome">Assistance Mode</span>
          </h1>
          
          <p className="text-xl text-muted-foreground max-w-2xl mx-auto font-light">
            Tailored intelligence for vehicle owners and professionals
          </p>
        </motion.div>

        {/* Mode Cards */}
        <div className="grid md:grid-cols-2 gap-12 max-w-6xl mx-auto">
          {/* OWNER MODE */}
          <motion.div
            custom={0}
            variants={cardVariants}
            initial="hidden"
            animate="visible"
            whileHover={{ y: -25, rotateY: 5, scale: 1.02 }}
            //onClick={() => navigate(`/chat?brand=${brand}&model=${model}&mode=owner`)}

            onClick={() => navigate(`/dashboard/owner?brand=${brand}&model=${model}`)}
            className="cursor-pointer"
            style={{ perspective: "1000px" }}
          >
            <Card className="relative overflow-hidden h-full p-12 glass-card border-2 border-transparent hover:border-secondary/50 transition-all duration-500 group">
              <div className="absolute inset-0 bg-gradient-silver opacity-0 group-hover:opacity-15 transition-opacity duration-500" />
              <div className="absolute inset-0 shadow-glow-silver opacity-0 group-hover:opacity-100 transition-opacity duration-500" />

              <div className="relative z-10 text-center">
                <div className="w-24 h-24 rounded-2xl bg-gradient-silver flex items-center justify-center mb-8 mx-auto group-hover:scale-110 transition-transform duration-500 shadow-elegant">
                  <User className="w-12 h-12 text-background" />
                </div>
                
                <h2 className="text-5xl font-bold mb-4 text-gradient-silver uppercase tracking-wider">Vehicle Owner</h2>
                <div className="w-24 h-0.5 bg-gradient-silver mx-auto mb-6 group-hover:w-32 transition-all duration-500" />
                
                <p className="text-muted-foreground text-lg leading-relaxed mb-8 font-light">
                  Personalized assistance for the discerning owner. Comprehensive vehicle knowledge, instant technical support, and emergency guidance.
                </p>
                
                <div className="space-y-3 mb-10 text-left">
                  {["Ask Technical Questions", "Explore Vehicle Features", "Maintenance Guidance", "Emergency Support"].map((feature, index) => (
                    <div key={index} className="flex items-center gap-3">
                      <div className="w-1.5 h-1.5 rounded-full bg-secondary" />
                      <span className="text-muted-foreground text-sm">{feature}</span>
                    </div>
                  ))}
                </div>
                
                <Button className="w-full bg-gradient-silver hover:bg-gradient-chrome text-background font-bold py-6 text-lg uppercase tracking-wider">
                  Enter Owner Mode
                </Button>
              </div>
            </Card>
          </motion.div>

          {/* MECHANIC MODE */}
          <motion.div
            custom={1}
            variants={cardVariants}
            initial="hidden"
            animate="visible"
            whileHover={{ y: -25, rotateY: -5, scale: 1.02 }}
           //onClick={() => navigate(`/chat?brand=${brand}&model=${model}&mode=mechanic`)}

            onClick={() => navigate(`/dashboard/mechanic?brand=${brand}&model=${model}`)}
            className="cursor-pointer"
            style={{ perspective: "1000px" }}
          >
            <Card className="relative overflow-hidden h-full p-12 glass-card border-2 border-transparent hover:border-accent-mechanic/50 transition-all duration-500 group">
              <div className="absolute inset-0 bg-gradient-gold opacity-0 group-hover:opacity-15 transition-opacity duration-500" />
              <div className="absolute inset-0 shadow-glow-gold opacity-0 group-hover:opacity-100 transition-opacity duration-500" />

              <div className="relative z-10 text-center">
                <div className="w-24 h-24 rounded-2xl bg-gradient-gold flex items-center justify-center mb-8 mx-auto group-hover:scale-110 transition-transform duration-500 shadow-elegant">
                  <Wrench className="w-12 h-12 text-background" />
                </div>
                
                <h2 className="text-5xl font-bold mb-4 text-gradient-gold uppercase tracking-wider">Professional Mechanic</h2>
                <div className="w-24 h-0.5 bg-gradient-gold mx-auto mb-6 group-hover:w-32 transition-all duration-500" />
                
                <p className="text-muted-foreground text-lg leading-relaxed mb-8 font-light">
                  Advanced diagnostic capabilities for automotive professionals. Technical precision, comprehensive reporting, and expert-level insights.
                </p>
                
                <div className="space-y-3 mb-10 text-left">
                  {["Advanced Diagnostics", "Technical Specifications", "Generate Service Reports", "Repair Procedures"].map((feature, index) => (
                    <div key={index} className="flex items-center gap-3">
                      <div className="w-1.5 h-1.5 rounded-full bg-accent-mechanic" />
                      <span className="text-muted-foreground text-sm">{feature}</span>
                    </div>
                  ))}
                </div>
                
                <Button className="w-full bg-gradient-gold hover:opacity-90 text-background font-bold py-6 text-lg uppercase tracking-wider">
                  Enter Mechanic Mode
                </Button>
              </div>
            </Card>
          </motion.div>
        </div>
      </div>
    </div>
  );
};

export default ModeSelection;
