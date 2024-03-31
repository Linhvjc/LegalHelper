import { Box, useMediaQuery, useTheme } from "@mui/material";
import React, { useEffect } from "react";
import TypingAnimAdmin from "../components/typer/TypingAnimAdmin";
import StickyHeadTableParam from "../components/table/StickyHeadTableParam";
import StickyHeadTable from "../components/table/StickyHeadTable";
import CollapsibleTable from "../components/table/CollapsibleTable";
import StickyHeadTableUser from "../components/table/StickyHeadTableUser";


const Admin = () => {
  const theme = useTheme();
  const isBelowMd = useMediaQuery(theme.breakpoints.down("md"));
  return (
    <Box width={"100%"} height={"100%"}>
      <Box
        sx={{
          display: "flex",
          width: "100%",
          flexDirection: "column",
          alignItems: "center",
          mx: "auto",
          mt: 3,
        }}
      >
        <Box>
          <TypingAnimAdmin />
        </Box>
        
        <Box
          sx={{
            width: "90%",
            display: "flex",
            flexDirection: { md: "row", xs: "column", sm: "column" },
            gap: 5,
            my: 10,
          }}
        >
          <StickyHeadTableUser />
        </Box>
        <Box
          sx={{
            width: "90%",
            display: "flex",
            flexDirection: { md: "row", xs: "column", sm: "column" },
            gap: 5,
            my: 10,
          }}
        >
          <StickyHeadTableParam />
        </Box>
        {/* <Box
          sx={{
            width: "80%",
            display: "flex",
            flexDirection: { md: "row", xs: "column", sm: "column" },
            gap: 5,
            my: 10,
          }}
        >
          <StickyHeadTable />
        </Box>
        <Box
          sx={{
            width: "80%",
            display: "flex",
            flexDirection: { md: "row", xs: "column", sm: "column" },
            gap: 5,
            my: 10,
          }}
        >
          <CollapsibleTable />
        </Box> */}
      </Box>
      {/* <Footer /> */}
    </Box>
  );
};

export default Admin;
