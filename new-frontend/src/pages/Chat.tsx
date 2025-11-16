// import { useState, useEffect, useRef } from "react";
// import { useNavigate, useSearchParams } from "react-router-dom";
// import { ArrowLeft, Send, User, Wrench, Loader2 } from "lucide-react";
// import { Button } from "@/components/ui/button";
// import { Textarea } from "@/components/ui/textarea";
// import { useToast } from "@/hooks/use-toast";

// interface Message {
//   role: "user" | "assistant";
//   content: string;
//   images?: string[];
//   tables?: string[];
// }

// const Chat = () => {
//   const navigate = useNavigate();
//   const [searchParams] = useSearchParams();
//   const mode = searchParams.get("mode") as "owner" | "mechanic" || "owner";
//   const { toast } = useToast();
  
//   const [messages, setMessages] = useState<Message[]>([]);
//   const [input, setInput] = useState("");
//   const [isLoading, setIsLoading] = useState(false);
//   const messagesEndRef = useRef<HTMLDivElement>(null);

//   const scrollToBottom = () => {
//     messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
//   };

//   useEffect(() => {
//     scrollToBottom();
//   }, [messages]);

//   const handleSubmit = async (e: React.FormEvent) => {
//     e.preventDefault();
//     if (!input.trim() || isLoading) return;

//     const userMessage: Message = { role: "user", content: input };
//     setMessages(prev => [...prev, userMessage]);
//     setInput("");
//     setIsLoading(true);

//     try {
//       const response = await fetch("http://localhost:8000/api/ask", {
//         method: "POST",
//         headers: {
//           "Content-Type": "application/json",
//         },
//         body: JSON.stringify({
//           query: input,
//           mode: mode,
//         }),
//       });

//       if (!response.ok) {
//         throw new Error("Failed to get response");
//       }

//       const data = await response.json();
//       const assistantMessage: Message = {
//         role: "assistant",
//         content: data.answer.answer_text || "I apologize, but I couldn't generate a response.",
//         images: data.answer.images || [],
//         tables: data.answer.tables || [],
//       };

//       setMessages(prev => [...prev, assistantMessage]);
//     } catch (error) {
//       console.error("Error:", error);
//       toast({
//         title: "Error",
//         description: "Failed to get response. Please make sure the backend server is running.",
//         variant: "destructive",
//       });
//     } finally {
//       setIsLoading(false);
//     }
//   };

//   const modeConfig = {
//     owner: {
//       icon: User,
//       color: "secondary",
//       gradient: "bg-gradient-silver",
//       glow: "shadow-glow-silver",
//       name: "Vehicle Owner",
//     },
//     mechanic: {
//       icon: Wrench,
//       color: "accent-mechanic",
//       gradient: "bg-gradient-gold",
//       glow: "shadow-glow-gold",
//       name: "Professional Mechanic",
//     },
//   };

//   const config = modeConfig[mode];
//   const ModeIcon = config.icon;

//   return (
//     <div className="min-h-screen bg-gradient-to-br from-background via-primary to-background flex flex-col">
//       {/* Header */}
//       <header className="border-b border-border bg-card/50 backdrop-blur-glass">
//         <div className="container mx-auto px-4 py-4 flex items-center justify-between">
//           <Button
//             variant="ghost"
//             size="sm"
//             onClick={() => navigate("/")}
//             className="gap-2"
//           >
//             <ArrowLeft className="w-4 h-4" />
//             Change Mode
//           </Button>
          
//           <div className="flex items-center gap-3">
//             <div className={`w-10 h-10 rounded-lg bg-${config.color}/10 flex items-center justify-center`}>
//               <ModeIcon className={`w-5 h-5 text-${config.color}`} />
//             </div>
//             <div>
//               <div className="text-sm font-medium">{config.name} Mode</div>
//               <div className="text-xs text-muted-foreground">AI Assistant</div>
//             </div>
//           </div>

//           <div className="w-24" /> {/* Spacer for centering */}
//         </div>
//       </header>

//       {/* Chat Area */}
//       <div className="flex-1 overflow-y-auto">
//         <div className="container mx-auto px-4 py-8 max-w-4xl">
//           {messages.length === 0 ? (
//             <div className="text-center py-16 animate-fade-in">
//               <div className={`w-20 h-20 mx-auto mb-6 rounded-2xl ${config.gradient} flex items-center justify-center ${config.glow}`}>
//                 <ModeIcon className="w-10 h-10 text-white" />
//               </div>
//               <h2 className="text-2xl font-bold mb-4">
//                 Welcome to {config.name} Mode
//               </h2>
//               <p className="text-muted-foreground max-w-md mx-auto">
//                 {mode === "owner" 
//                   ? "Ask me anything about your vehicle. I'll provide clear, easy-to-understand answers."
//                   : "Get detailed technical specifications and professional diagnostic information for your repairs."}
//               </p>
//             </div>
//           ) : (
//             <div className="space-y-6">
//               {messages.map((message, index) => (
//                 <div
//                   key={index}
//                   className={`flex gap-4 animate-fade-in-up ${
//                     message.role === "user" ? "justify-end" : "justify-start"
//                   }`}
//                 >
//                   {message.role === "assistant" && (
//                     <div className={`w-10 h-10 rounded-lg ${config.gradient} flex items-center justify-center flex-shrink-0`}>
//                       <ModeIcon className="w-5 h-5 text-white" />
//                     </div>
//                   )}
                  
//                   <div
//                     className={`max-w-2xl rounded-2xl p-6 ${
//                       message.role === "user"
//                         ? "bg-muted"
//                         : "bg-card/50 backdrop-blur-glass border border-border"
//                     }`}
//                   >
//                     <p className="whitespace-pre-wrap leading-relaxed">{message.content}</p>
                    
//                     {message.images && message.images.length > 0 && (
//                       <div className="mt-4 grid grid-cols-2 gap-2">
//                         {message.images.map((img, i) => (
//                           <img
//                             key={i}
//                             src={img}
//                             alt={`Diagram ${i + 1}`}
//                             className="rounded-lg border border-border"
//                           />
//                         ))}
//                       </div>
//                     )}
                    
//                     {message.tables && message.tables.length > 0 && (
//                       <div className="mt-4 space-y-2">
//                         {message.tables.map((table, i) => (
//                           <div key={i} className="p-4 bg-background/50 rounded-lg text-sm font-mono">
//                             {table}
//                           </div>
//                         ))}
//                       </div>
//                     )}
//                   </div>

//                   {message.role === "user" && (
//                     <div className="w-10 h-10 rounded-lg bg-muted flex items-center justify-center flex-shrink-0">
//                       <User className="w-5 h-5" />
//                     </div>
//                   )}
//                 </div>
//               ))}
              
//               {isLoading && (
//                 <div className="flex gap-4 animate-fade-in">
//                   <div className={`w-10 h-10 rounded-lg ${config.gradient} flex items-center justify-center flex-shrink-0`}>
//                     <ModeIcon className="w-5 h-5 text-white" />
//                   </div>
//                   <div className="max-w-2xl rounded-2xl p-6 bg-card/50 backdrop-blur-glass border border-border">
//                     <div className="flex items-center gap-2">
//                       <Loader2 className="w-4 h-4 animate-spin" />
//                       <span className="text-muted-foreground">Analyzing your question...</span>
//                     </div>
//                   </div>
//                 </div>
//               )}
              
//               <div ref={messagesEndRef} />
//             </div>
//           )}
//         </div>
//       </div>

//       {/* Input Area */}
//       <div className="border-t border-border bg-card/50 backdrop-blur-glass">
//         <div className="container mx-auto px-4 py-6 max-w-4xl">
//           <form onSubmit={handleSubmit} className="flex gap-4">
//             <Textarea
//               value={input}
//               onChange={(e) => setInput(e.target.value)}
//               placeholder={mode === "owner" 
//                 ? "Ask about your vehicle..." 
//                 : "Enter technical query..."}
//               className="flex-1 min-h-[60px] max-h-[200px] bg-background/50 border-border focus:ring-2 focus:ring-accent resize-none"
//               onKeyDown={(e) => {
//                 if (e.key === "Enter" && !e.shiftKey) {
//                   e.preventDefault();
//                   handleSubmit(e);
//                 }
//               }}
//             />
//             <Button
//               type="submit"
//               disabled={isLoading || !input.trim()}
//               className={`${config.gradient} hover:opacity-90 ${config.glow} h-[60px] px-8 text-background font-semibold`}
//             >
//               {isLoading ? (
//                 <Loader2 className="w-5 h-5 animate-spin" />
//               ) : (
//                 <>
//                   <Send className="w-5 h-5 mr-2" />
//                   Ask
//                 </>
//               )}
//             </Button>
//           </form>
//         </div>
//       </div>
//     </div>
//   );
// };

// export default Chat;
import { useState, useEffect, useRef } from "react";
import { useNavigate, useSearchParams } from "react-router-dom";
import { ArrowLeft, Send, User, Wrench, Loader2, Car } from "lucide-react";

interface Message {
  role: "user" | "assistant";
  content: string;
  images?: string[];
  tables?: string[];
}

const Chat = () => {
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();
  const mode = searchParams.get("mode") as "owner" | "mechanic" || "owner";
  const brand = searchParams.get("brand") || "";
  const model = searchParams.get("model") || "";
  
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // Redirect if no car selected
  useEffect(() => {
    if (!brand || !model) {
      navigate("/brand-selection");
    }
  }, [brand, model, navigate]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || isLoading) return;

    const userMessage: Message = { role: "user", content: input };
    setMessages(prev => [...prev, userMessage]);
    setInput("");
    setIsLoading(true);

    try {
      const response = await fetch("http://localhost:8000/api/ask", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          query: input,
          manufacturer: brand,
          model: model,
          mode: mode,
        }),
      });

      if (!response.ok) {
        throw new Error("Failed to get response");
      }

      const data = await response.json();
      const assistantMessage: Message = {
        role: "assistant",
        content: data.answer.answer_text || "I apologize, but I couldn't generate a response.",
        images: data.answer.images || [],
        tables: data.answer.tables || [],
      };

      setMessages(prev => [...prev, assistantMessage]);
    } catch (error) {
      console.error("Error:", error);
      const errorMessage: Message = {
        role: "assistant",
        content: "Failed to get response. Please make sure the backend server is running.",
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const modeConfig = {
    owner: {
      icon: User,
      color: "secondary",
      gradient: "bg-gradient-to-br from-blue-500 to-blue-600",
      glow: "shadow-lg shadow-blue-500/50",
      name: "Vehicle Owner",
    },
    mechanic: {
      icon: Wrench,
      color: "accent-mechanic",
      gradient: "bg-gradient-to-br from-orange-500 to-orange-600",
      glow: "shadow-lg shadow-orange-500/50",
      name: "Professional Mechanic",
    },
  };

  const config = modeConfig[mode];
  const ModeIcon = config.icon;

  // Format car name
  const carName = `${brand.charAt(0).toUpperCase() + brand.slice(1)} ${model.split('-').map(w => w.charAt(0).toUpperCase() + w.slice(1)).join(' ')}`;

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 flex flex-col">
      {/* Header */}
      <header className="border-b border-slate-700 bg-slate-800/50 backdrop-blur-sm">
        <div className="container mx-auto px-4 py-4 flex items-center justify-between">
          <button
            onClick={() => navigate(`/mode-selection?brand=${brand}&model=${model}`)}
            className="flex items-center gap-2 px-4 py-2 rounded-lg hover:bg-slate-700/50 transition-colors"
          >
            <ArrowLeft className="w-4 h-4" />
            Change Mode
          </button>
          
          <div className="flex items-center gap-4">
            <div className="flex items-center gap-2 px-4 py-2 rounded-lg bg-slate-700/50">
              <Car className="w-4 h-4 text-blue-400" />
              <span className="text-sm font-medium">{carName}</span>
            </div>
            
            <div className={`flex items-center gap-3 px-4 py-2 rounded-lg ${config.gradient}`}>
              <ModeIcon className="w-5 h-5 text-white" />
              <div>
                <div className="text-sm font-medium text-white">{config.name}</div>
              </div>
            </div>
          </div>

          <div className="w-32" />
        </div>
      </header>

      {/* Chat Area */}
      <div className="flex-1 overflow-y-auto">
        <div className="container mx-auto px-4 py-8 max-w-4xl">
          {messages.length === 0 ? (
            <div className="text-center py-16 animate-fade-in">
              <div className={`w-20 h-20 mx-auto mb-6 rounded-2xl ${config.gradient} flex items-center justify-center ${config.glow}`}>
                <ModeIcon className="w-10 h-10 text-white" />
              </div>
              <h2 className="text-2xl font-bold mb-4 text-white">
                Welcome to {config.name} Mode
              </h2>
              <p className="text-slate-400 max-w-md mx-auto mb-4">
                {mode === "owner" 
                  ? "Ask me anything about your vehicle. I'll provide clear, easy-to-understand answers."
                  : "Get detailed technical specifications and professional diagnostic information for your repairs."}
              </p>
              <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-slate-700/50 text-sm text-slate-300">
                <Car className="w-4 h-4" />
                Searching in: {carName} Manual
              </div>
            </div>
          ) : (
            <div className="space-y-6">
              {messages.map((message, index) => (
                <div
                  key={index}
                  className={`flex gap-4 ${
                    message.role === "user" ? "justify-end" : "justify-start"
                  }`}
                >
                  {message.role === "assistant" && (
                    <div className={`w-10 h-10 rounded-lg ${config.gradient} flex items-center justify-center flex-shrink-0`}>
                      <ModeIcon className="w-5 h-5 text-white" />
                    </div>
                  )}
                  
                  <div
                    className={`max-w-2xl rounded-2xl p-6 ${
                      message.role === "user"
                        ? "bg-slate-700"
                        : "bg-slate-800 border border-slate-700"
                    }`}
                  >
                    <p className="whitespace-pre-wrap leading-relaxed text-slate-100">{message.content}</p>
                    
                    {message.images && message.images.length > 0 && (
                      <div className="mt-4 grid grid-cols-2 gap-2">
                        {message.images.map((img, i) => (
                          <img
                            key={i}
                            src={img}
                            alt={`Diagram ${i + 1}`}
                            className="rounded-lg border border-slate-600"
                          />
                        ))}
                      </div>
                    )}
                    
                    {message.tables && message.tables.length > 0 && (
                      <div className="mt-4 space-y-2">
                        {message.tables.map((table, i) => (
                          <div key={i} className="p-4 bg-slate-900/50 rounded-lg text-sm font-mono text-slate-300">
                            {table}
                          </div>
                        ))}
                      </div>
                    )}
                  </div>

                  {message.role === "user" && (
                    <div className="w-10 h-10 rounded-lg bg-slate-700 flex items-center justify-center flex-shrink-0">
                      <User className="w-5 h-5 text-slate-300" />
                    </div>
                  )}
                </div>
              ))}
              
              {isLoading && (
                <div className="flex gap-4">
                  <div className={`w-10 h-10 rounded-lg ${config.gradient} flex items-center justify-center flex-shrink-0`}>
                    <ModeIcon className="w-5 h-5 text-white" />
                  </div>
                  <div className="max-w-2xl rounded-2xl p-6 bg-slate-800 border border-slate-700">
                    <div className="flex items-center gap-2">
                      <Loader2 className="w-4 h-4 animate-spin text-blue-400" />
                      <span className="text-slate-400">Searching {carName} manual...</span>
                    </div>
                  </div>
                </div>
              )}
              
              <div ref={messagesEndRef} />
            </div>
          )}
        </div>
      </div>

      {/* Input Area */}
      <div className="border-t border-slate-700 bg-slate-800/50 backdrop-blur-sm">
        <div className="container mx-auto px-4 py-6 max-w-4xl">
          <form onSubmit={handleSubmit} className="flex gap-4">
            <textarea
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder={mode === "owner" 
                ? "Ask about your vehicle..." 
                : "Enter technical query..."}
              className="flex-1 min-h-[60px] max-h-[200px] bg-slate-900 border border-slate-700 rounded-lg px-4 py-3 text-slate-100 placeholder-slate-500 focus:ring-2 focus:ring-blue-500 focus:border-transparent resize-none"
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  handleSubmit(e);
                }
              }}
            />
            <button
              type="submit"
              disabled={isLoading || !input.trim()}
              className={`${config.gradient} hover:opacity-90 ${config.glow} h-[60px] px-8 rounded-lg text-white font-semibold disabled:opacity-50 disabled:cursor-not-allowed transition-all`}
            >
              {isLoading ? (
                <Loader2 className="w-5 h-5 animate-spin" />
              ) : (
                <div className="flex items-center gap-2">
                  <Send className="w-5 h-5" />
                  Ask
                </div>
              )}
            </button>
          </form>
        </div>
      </div>
    </div>
  );
};

export default Chat;