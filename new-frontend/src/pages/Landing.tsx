import { useNavigate } from "react-router-dom";
import { User, Wrench, Sparkles } from "lucide-react";
import { Button } from "@/components/ui/button";

const Landing = () => {
  const navigate = useNavigate();

  const handleModeSelect = (mode: "owner" | "mechanic") => {
    navigate(`/chat?mode=${mode}`);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-background via-primary to-background flex items-center justify-center p-4 relative overflow-hidden">
      
      <div className="absolute inset-0 overflow-hidden">
        <div className="absolute top-1/4 left-1/4 w-96 h-96 bg-accent-owner/10 rounded-full blur-3xl animate-float" />
        <div className="absolute bottom-1/4 right-1/4 w-96 h-96 bg-accent-mechanic/10 rounded-full blur-3xl animate-float" style={{ animationDelay: "1.5s" }} />
      </div>

      <div className="relative z-10 max-w-6xl w-full animate-fade-in-up">
       
        <div className="text-center mb-16">
          <div className="inline-flex items-center gap-2 mb-6 px-4 py-2 rounded-full bg-card/50 backdrop-blur-glass border border-border">
            <Sparkles className="w-4 h-4 text-accent-owner" />
            <span className="text-sm font-medium text-muted-foreground">AI-Powered Automotive Intelligence</span>
          </div>
          <h1 className="text-5xl md:text-7xl font-bold mb-6 bg-gradient-to-r from-foreground via-secondary to-foreground bg-clip-text text-transparent">
            Intelligent Automotive Assistant
          </h1>
          <p className="text-xl text-muted-foreground max-w-2xl mx-auto">
            Get instant, expert answers from your vehicle's technical manual with AI precision
          </p>
        </div>

       
        <div className="grid md:grid-cols-2 gap-8 max-w-4xl mx-auto">
          
          <div 
            className="group relative overflow-hidden rounded-2xl bg-card/50 backdrop-blur-glass border border-border p-8 transition-all duration-300 hover:scale-105 hover:shadow-glow-owner cursor-pointer"
            onClick={() => handleModeSelect("owner")}
          >
            <div className="absolute inset-0 bg-gradient-owner opacity-0 group-hover:opacity-10 transition-opacity duration-300" />
            <div className="absolute inset-0 border-2 border-accent-owner/0 group-hover:border-accent-owner/50 rounded-2xl transition-all duration-300" />
            
            <div className="relative z-10">
              <div className="w-16 h-16 rounded-xl bg-accent-owner/10 flex items-center justify-center mb-6 group-hover:bg-accent-owner/20 transition-colors duration-300">
                <User className="w-8 h-8 text-accent-owner" />
              </div>
              
              <h3 className="text-2xl font-bold mb-4 group-hover:text-accent-owner transition-colors duration-300">
                Vehicle Owner Mode
              </h3>
              
              <p className="text-muted-foreground mb-6 leading-relaxed">
                Simple, easy-to-understand answers for everyday vehicle questions. Perfect for maintenance, troubleshooting, and understanding your car better.
              </p>
              
              <Button 
                variant="outline" 
                className="w-full group-hover:bg-accent-owner group-hover:text-accent-owner-foreground group-hover:border-accent-owner transition-all duration-300"
              >
                Get Started
              </Button>
            </div>
          </div>

          
          <div 
            className="group relative overflow-hidden rounded-2xl bg-card/50 backdrop-blur-glass border border-border p-8 transition-all duration-300 hover:scale-105 hover:shadow-glow-mechanic cursor-pointer"
            onClick={() => handleModeSelect("mechanic")}
          >
            <div className="absolute inset-0 bg-gradient-mechanic opacity-0 group-hover:opacity-10 transition-opacity duration-300" />
            <div className="absolute inset-0 border-2 border-accent-mechanic/0 group-hover:border-accent-mechanic/50 rounded-2xl transition-all duration-300" />
            
            <div className="relative z-10">
              <div className="w-16 h-16 rounded-xl bg-accent-mechanic/10 flex items-center justify-center mb-6 group-hover:bg-accent-mechanic/20 transition-colors duration-300">
                <Wrench className="w-8 h-8 text-accent-mechanic" />
              </div>
              
              <h3 className="text-2xl font-bold mb-4 group-hover:text-accent-mechanic transition-colors duration-300">
                Professional Mechanic Mode
              </h3>
              
              <p className="text-muted-foreground mb-6 leading-relaxed">
                Technical specifications, diagnostic procedures, and detailed repair information. Designed for automotive professionals and technicians.
              </p>
              
              <Button 
                variant="outline" 
                className="w-full group-hover:bg-accent-mechanic group-hover:text-background group-hover:border-accent-mechanic transition-all duration-300"
              >
                Get Started
              </Button>
            </div>
          </div>
        </div>

       
        <div className="text-center mt-12 text-sm text-muted-foreground">
          Powered by advanced AI trained on comprehensive automotive technical manuals
        </div>
      </div>
    </div>
  );
};

export default Landing;
